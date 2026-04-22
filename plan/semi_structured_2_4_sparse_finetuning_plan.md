# Semi-Structured 2:4 Sparse Fine-Tuning Plan

## Goal

Sparse fine-tune all three models — **Llama 3.1 8B Instruct**, **Qwen3 8B**, and **Gemma4 E4B IT** — on MMLU after 2:4 semi-structured pruning, preserving the 2:4 base-weight mask throughout training. All runs log to the W&B project **`sparse-sft`** (already configured in `configs/finetuning/structured_lora_recovery.yaml`).

Primary comparison points:

* Dense baseline: `0%`
* Prune-only semi-structured `2:4`
* Prune plus sparse fine-tuning semi-structured `2:4`

The first pass uses the repository's existing semi-structured magnitude path:

```yaml
pruning:
  method: global_magnitude_semi_structured
  structure: nm_2_4
  nm_n: 2
  nm_m: 4
  block_dim: 1
  sparsities: [0, 50]
```

---

## Current Repo State

### Already implemented and reusable

| Path | Purpose |
|------|---------|
| `configs/pruning/global_semi_structured_2_4.yaml` | 2:4 pruning config |
| `src/llm_pruning_mmlu/pruning/semi_structured.py` | N:M mask generation and validation |
| `src/llm_pruning_mmlu/pruning/dispatch.py` | Dispatcher — returns masks for semi-structured pruning |
| `src/llm_pruning_mmlu/finetuning/mask_policy.py` | `MaskEnforcerCallback` — reapplies base masks after every optimizer step |
| `src/llm_pruning_mmlu/finetuning/runner.py` | SFT sweep runner |
| `src/llm_pruning_mmlu/finetuning/masked_trainer.py` | Trainer with mask enforcement |
| `src/llm_pruning_mmlu/finetuning/lora.py` | LoRA setup |
| `src/llm_pruning_mmlu/finetuning/wandb_utils.py` | W&B integration |
| `src/llm_pruning_mmlu/finetuning/datasets.py` | Dataset loading |
| `src/llm_pruning_mmlu/finetuning/config.py` | Finetuning config dataclass |
| `scripts/run_sparse_finetune.py` | CLI entry point for SFT runs |
| `configs/finetuning/structured_lora_recovery.yaml` | LoRA config with `wandb.project: sparse-sft` |
| `slurm/sparse_sft/structured/` | Structured SFT Slurm scripts — copy these as templates |
| `configs/experiments/all_models_structured_sft.yaml` | Structured SFT all-model config — copy pattern for semi-structured |

### Not yet created (required for this plan)

| Path | Needed for |
|------|-----------|
| `configs/experiments/all_models_semi_structured_2_4_sft.yaml` | Phase 1 |
| `configs/experiments/llama31_semi_structured_2_4_sft.yaml` | Phase 1 |
| `configs/experiments/qwen3_semi_structured_2_4_sft.yaml` | Phase 1 |
| `configs/experiments/gemma4_e4b_semi_structured_2_4_sft.yaml` | Phase 1 |
| `slurm/sparse_sft/semi_structured/run_llama31_2_4_sft.slurm` | Phase 4 |
| `slurm/sparse_sft/semi_structured/run_qwen3_2_4_sft.slurm` | Phase 4 |
| `slurm/sparse_sft/semi_structured/run_gemma4_2_4_sft.slurm` | Phase 4 |
| `scripts/validate_sparse_sft_masks.py` | Phase 5 |

> **Note**: The existing `configs/experiments/*mmlu_semi_structured_2_4_sweep.yaml` files are
> pruning-only sweep configs — they have no `output_root: outputs/sft_runs` and are not suitable
> as SFT experiment configs. Create the SFT configs separately following the pattern from
> `configs/experiments/all_models_structured_sft.yaml`.

### Known bugs to fix before running

**Bug 1 — `--max-eval-samples` controls final eval, not SFT validation**
In `scripts/run_sparse_finetune.py` line 50:
```python
exp_dict.setdefault("evaluation", {})["max_samples"] = args.max_eval_samples
```
This sets the final MMLU evaluation sample count, **not** `ft_dict["finetuning"]["validation_dataset"]["max_samples"]`. The CLI flag is misleading for smoke tests. To cap SFT validation samples, edit `validation_dataset.max_samples` directly in the finetuning config, or fix the script to also set:
```python
ft_dict.setdefault("finetuning", {}).setdefault("validation_dataset", {})["max_samples"] = args.max_eval_samples
```

**Bug 2 — `--sparsities` parses as float, config expects int**
`argparse` uses `type=float`, so `--sparsities 50` becomes `[50.0]`. The Phase 1 acceptance check compares `cfg.pruning.sparsities == [0, 50]` (int list). Verify `parse_config` coerces `[50.0]` → `[50]`; if not, change `type=float` to `type=int` in the argument definition or add an explicit cast.

**Bug 3 — Smoke tests log to W&B by default**
`configs/finetuning/structured_lora_recovery.yaml` has `wandb.enabled: true`. Every `--fail-fast` smoke run will create a W&B run. Either add a `--no-wandb` flag to `run_sparse_finetune.py`:
```python
parser.add_argument("--no-wandb", action="store_true")
# then: ft_dict.setdefault("finetuning", {}).setdefault("wandb", {})["enabled"] = False
```
or temporarily set `enabled: false` in the finetuning config before smoke tests.

---

## Phase 1: Add 2:4 SFT Experiment Configs

**Status: TODO**

Create the all-model SFT config following the same pattern as `configs/experiments/all_models_structured_sft.yaml`:

```yaml
# configs/experiments/all_models_semi_structured_2_4_sft.yaml
inherits:
  - configs/base.yaml
  - configs/datasets/mmlu.yaml
  - configs/pruning/global_semi_structured_2_4.yaml

output_root: outputs/sft_runs

models:
  - configs/models/llama31_8b_instruct.yaml
  - configs/models/qwen3_8b.yaml
  - configs/models/gemma4_e4b_it.yaml
```

Create per-model configs (same inherits, single model each):

```yaml
# configs/experiments/qwen3_semi_structured_2_4_sft.yaml
inherits:
  - configs/base.yaml
  - configs/datasets/mmlu.yaml
  - configs/pruning/global_semi_structured_2_4.yaml

output_root: outputs/sft_runs

models:
  - configs/models/qwen3_8b.yaml
```

Repeat for llama31 and gemma4.

Acceptance check:

```bash
python - <<'PY'
from llm_pruning_mmlu.config import load_config_dict, parse_config
cfg = parse_config(load_config_dict("configs/experiments/all_models_semi_structured_2_4_sft.yaml"))
assert cfg.pruning.method == "global_magnitude_semi_structured"
assert cfg.pruning.nm_n == 2
assert cfg.pruning.nm_m == 4
assert cfg.output_root == "outputs/sft_runs"
print("ok")
PY
```

> Note: removed the `cfg.pruning.sparsities == [0, 50]` assertion pending Bug 2 fix above.

---

## Phase 2: Finetuning Config — Already Done

**Status: DONE** — `configs/finetuning/structured_lora_recovery.yaml` is ready as-is.

Full config for reference (do not modify unless tuning hyperparameters):

```yaml
finetuning:
  method: lora
  train_dataset:
    hf_id: cais/mmlu
    split: auxiliary_train
    max_samples: 10000
  validation_dataset:
    hf_id: cais/mmlu
    split: validation
    max_samples: 512
  max_seq_length: 1024
  epochs: 1
  learning_rate: 0.0002
  batch_size: 1                      # effective batch = 1 * 16 = 16
  gradient_accumulation_steps: 16
  warmup_ratio: 0.03
  weight_decay: 0.0
  bf16: true
  gradient_checkpointing: true
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj
  mask_policy:
    preserve_base_masks: true
    mask_lora_pruned_channels: false
  wandb:
    project: sparse-sft
    enabled: true
```

Do not train on the MMLU test split:

* Train: `auxiliary_train`
* Validation during SFT: `validation`
* Final evaluation: evaluation split from the experiment config

---

## Phase 3: Local Smoke Test

**Status: TODO** (depends on Phase 1 + Bug fixes)

Fix Bug 1, Bug 2, and Bug 3 first, then run:

```bash
python scripts/run_sparse_finetune.py \
  --config configs/experiments/qwen3_semi_structured_2_4_sft.yaml \
  --finetune-config configs/finetuning/structured_lora_recovery.yaml \
  --sparsities 50 \
  --max-train-samples 32 \
  --max-eval-samples 16 \
  --no-wandb \
  --no-resume \
  --fail-fast
```

Expected artifacts under `outputs/sft_runs/<run_id>/.../sparsity_050/`:

```
adapter/
masks.pt
metrics.json
predictions.jsonl
pruning_stats.json
resolved_config.yaml
run.log
training_stats.json
```

Minimum smoke-test checks:

* `metrics.json` has `pruning_method = global_magnitude_semi_structured`
* `metrics.json` has `pruning_kind = semi_structured`
* `sparsity_requested = 50`, `sparsity_achieved` ≈ 50%
* `masks.pt` exists
* `training_stats.json` has `mask_enforcer_active = true`
* `run.log` has no N:M validation failure

---

## Phase 4: Add Slurm Entry Points

**Status: TODO**

Copy the pattern from `slurm/sparse_sft/structured/run_qwen3_structured_sft.slurm` to create:

```
slurm/sparse_sft/semi_structured/run_llama31_2_4_sft.slurm
slurm/sparse_sft/semi_structured/run_qwen3_2_4_sft.slurm
slurm/sparse_sft/semi_structured/run_gemma4_2_4_sft.slurm
```

Required defaults in each script:

```bash
CONFIG="${CONFIG:-configs/experiments/<model>_semi_structured_2_4_sft.yaml}"
FINETUNE_CONFIG="${FINETUNE_CONFIG:-configs/finetuning/structured_lora_recovery.yaml}"
```

Slurm smoke submission (Qwen first):

```bash
sbatch --export=ALL,MAX_TRAIN_SAMPLES=128,MAX_EVAL_SAMPLES=64,SPARSITIES="50",FAIL_FAST=1,NO_RESUME=1 \
  slurm/sparse_sft/semi_structured/run_qwen3_2_4_sft.slurm
```

Full per-model submissions:

```bash
sbatch --export=ALL,SPARSITIES="50" slurm/sparse_sft/semi_structured/run_llama31_2_4_sft.slurm
sbatch --export=ALL,SPARSITIES="50" slurm/sparse_sft/semi_structured/run_qwen3_2_4_sft.slurm
sbatch --export=ALL,SPARSITIES="50" slurm/sparse_sft/semi_structured/run_gemma4_2_4_sft.slurm
```

Run sparsity `0` through SFT only if a LoRA-trained dense baseline is needed; otherwise use existing dense results and run only `50`.

---

## Phase 5: Correctness Validation

**Status: TODO** — `scripts/validate_sparse_sft_masks.py` does not exist yet.

Create the script. Required checks:

1. Load `masks.pt`.
2. Load the final trained model + adapter state.
3. For each masked base parameter, assert every `False` mask position is still zero.
4. Re-run `validate_nm_mask(parameter, mask, n=2, m=4, block_dim=1)` for every masked parameter.
5. Write validation status to `mask_validation.json` in the run directory.

CLI:

```bash
python scripts/validate_sparse_sft_masks.py \
  --run-dir outputs/sft_runs/<run_id>/<model>/global_magnitude_semi_structured__nm_2_4/sparsity_050 \
  --config configs/experiments/qwen3_semi_structured_2_4_sft.yaml
```

Acceptance criteria:

* No pruned base weight has regrown.
* Every complete block has exactly 2 zeros per 4 weights.
* Remainder columns (if any) are not pruned.
* `lm_head` is not pruned.

---

## Phase 6: Results Table

**Status: TODO**

For each of the three models, report:

| Column | Description |
|--------|-------------|
| `dense_accuracy` | Baseline MMLU accuracy at 0% sparsity |
| `prune_only_accuracy` | 2:4 prune-only MMLU accuracy |
| `sft_accuracy` | 2:4 + LoRA sparse SFT MMLU accuracy |
| `accuracy_recovered` | `sft_accuracy - prune_only_accuracy` |
| `recovery_fraction` | `(sft - prune) / (dense - prune)` |
| `sparsity_achieved` | Actual parameter sparsity |
| `mask_valid` | Whether Phase 5 validation passed |
| `train_samples` | Samples used for SFT |
| `eval_samples` | Samples used for final eval |
| `runtime_s` | Wall-clock training time |

Output target:

```
outputs/summaries/semi_structured_2_4_sparse_sft_summary.csv
```

---

## Phase 7: Follow-Up Wanda-Style 2:4 Masks

**Status: IMPLEMENTED** — code is in place; run after Phase 6 results are stable.

Add `wanda_semi_structured` scoring:

```yaml
pruning:
  method: wanda_semi_structured
  structure: nm_2_4
  nm_n: 2
  nm_m: 4
```

Scoring formula:

```
score[row, col] = abs(weight[row, col]) * sqrt(input_activation_norm[col])
```

Reuse the exact same SFT runner and mask-preservation path. Enables a clean four-way comparison:

* Magnitude 2:4 prune-only
* Magnitude 2:4 + SFT
* Wanda 2:4 prune-only
* Wanda 2:4 + SFT

---

## Done Criteria

* [x] Bug 1, Bug 2, Bug 3 fixed in `scripts/run_sparse_finetune.py`
* [x] 2:4 SFT experiment configs created — magnitude and Wanda (Phase 1 + Phase 7)
* [x] Slurm scripts created for all three models — magnitude and Wanda (Phase 4 + Phase 7)
* [x] `scripts/validate_sparse_sft_masks.py` created (Phase 5)
* [x] `scripts/summarize_sparse_sft.py` created (Phase 6)
* [x] `src/llm_pruning_mmlu/pruning/wanda.py` implemented (Phase 7)
* [x] `wanda_semi_structured` wired into dispatch + config (Phase 7)
* [ ] Acceptance check passes (`parse_config` round-trip on all_models config)
* [ ] Smoke run completes with `--fail-fast` for Qwen (Phase 3)
* [ ] `masks.pt`, adapter, metrics, and training stats written
* [ ] Full 50% 2:4 SFT runs complete for Llama, Qwen, and Gemma
* [ ] Mask validation confirms no base-weight regrowth (Phase 5)
* [ ] Summary CSV written comparing dense, prune-only, and prune+SFT (Phase 6)
* [ ] Final notes state this is masked-dense execution — sparse kernels require a separate runtime conversion step
