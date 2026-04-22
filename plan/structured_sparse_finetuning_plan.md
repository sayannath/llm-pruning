# Structured Sparse Fine-Tuning Plan

## Goal

Fine-tune structured-pruned Llama, Qwen, and Gemma models on MMLU to recover MMLU accuracy after MLP-channel pruning.

Primary sparsity points:

* `10%`
* `20%`
* `30%`
* `40%`
* `50%`

The first implementation should use the existing structured pruning path:

```text
method: global_magnitude_structured
structure: mlp_channel
score: l2_norm
scope: global
```

The fine-tuning stage should preserve the structured mask during training so pruned MLP channels do not regrow. The output should make it easy to compare:

* dense baseline
* prune-only structured sparsity
* prune plus sparse fine-tuning

---

## Experiment Scope

### Models

Run the structured sparse fine-tuning sweep for all three model families:

```text
configs/models/llama31_8b_instruct.yaml
configs/models/qwen3_8b.yaml
configs/models/gemma4_e4b_it.yaml
```

### Sparsity Grid

Use only the recovery-relevant structured sparsity points:

```yaml
sparsities: [10, 20, 30, 40, 50]
```

Keep the existing prune-only structured sweep results as the baseline comparison. Do not include `60` or `70` in the initial sparse fine-tuning run unless the 50% model recovers well.

### Training Dataset

Use MMLU for sparse fine-tuning and MMLU for evaluation, with strict split separation.

Recommended split usage:

* Training: `cais/mmlu`, `auxiliary_train` split if available in the local loader/runtime.
* Validation during fine-tuning: `cais/mmlu`, `validation` or `dev` split.
* Final evaluation: `cais/mmlu`, `test` split.

Do not fine-tune on MMLU test data. If the current MMLU loader only supports the test split, add split support before implementing training.

---

## Method

### Sparse Fine-Tuning Flow

For each sparsity point:

1. Load the dense base model.
2. Apply structured MLP-channel pruning at the requested sparsity.
3. **Extend `prune_model` (or `_prune_structured`) to return the `masks` dict in addition to `(parameters, stats)`.** Save `masks.pt` immediately after pruning. The current `dispatch.py::_prune_structured` discards masks after calling `apply_masks` — this must be changed before `masks.pt` can be written.
4. Attach LoRA adapters to trainable projection modules.
5. During training, keep the pruned structured channels inactive (see Mask Preservation below).
6. Save the adapter, pruning mask metadata, resolved config, and training metrics.
7. Reload the pruned model plus adapter.
8. Evaluate on MMLU with the current `evaluate_examples` path.

### Why LoRA First

LoRA is the safest first fine-tuning route for the current codebase:

* It avoids full-model optimizer memory on H100 jobs.
* It keeps the base pruned model stable.
* It creates small artifacts per sparsity point.
* It allows adapter-only comparison across sparsities.

Add dependencies only when implementing:

```text
peft
trl
wandb
```

If full sparse fine-tuning is needed later, implement it as a second phase after the LoRA recovery experiment has clear results.

### Mask Preservation

The key correctness requirement is that pruned channels stay pruned after optimization.

For MLP-channel structured pruning, one group spans:

```text
gate_proj.weight[channel, :]
up_proj.weight[channel, :]
down_proj.weight[:, channel]
```

At minimum, enforce masks after every optimizer step:

```text
optimizer.step()
apply_masks(parameters, masks)   # NOTE: signature is apply_masks(parameters, masks), NOT (model, masks)
optimizer.zero_grad()
```

The `PruningParameter` list returned by step 3 must be kept alive for the entire training loop — do not set it to `None` until after training completes. This is the same pattern the sweep runner follows (`targets = None` only in `finally`).

If LoRA is attached to the same modules, also decide whether LoRA should be masked:

* First run: allow LoRA to adapt dense dimensions, but keep base pruned weights masked.
* Stricter follow-up: mask LoRA outputs for pruned channels as well, so adapters cannot route around removed groups.

Record which behavior was used in `training_stats.json`.

---

## New Files To Add

### Configs

Add a pruning config with only the desired sparse fine-tuning sparsities:

```text
configs/pruning/global_structured_mlp_sft.yaml
```

Expected content:

```yaml
pruning:
  method: global_magnitude_structured
  structure: mlp_channel
  score: l2_norm
  scope: global
  target_module_types:
    - Linear
  target_parameter_names:
    - weight
  exclude_module_name_patterns:
    - lm_head
  prune_bias: false
  sparsities: [10, 20, 30, 40, 50]
```

Add fine-tuning config support:

```text
configs/finetuning/structured_lora_recovery.yaml
```

Suggested fields:

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
  batch_size: 1
  gradient_accumulation_steps: 16
  warmup_ratio: 0.03
  weight_decay: 0.0
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

The `FinetuningConfig` dataclass in `finetuning/config.py` should be a **standalone** config class, not a field on `ExperimentConfig`. The SFT runner reads two separate config files:

* The existing experiment YAML (model, device, pruning, dataset/eval) via `load_config`.
* The finetuning YAML parsed into `FinetuningConfig`.

Do **not** add a `finetuning` field to `ExperimentConfig` — that would couple the pruning config system to training and break all existing sweep runners.

> **Note on MMLU splits**: `load_mmlu` already accepts any `split` string and passes it directly to `load_dataset`. No changes to the loader are required; `auxiliary_train`, `validation`, and `test` all work today.

Add model-level experiment configs:

```text
configs/experiments/all_models_structured_sft.yaml
configs/experiments/gemma4_e4b_structured_sft.yaml
configs/experiments/llama31_structured_sft.yaml
configs/experiments/qwen3_structured_sft.yaml
```

### Python Modules

The fine-tuning package is implemented. All modules are **pruning-kind-agnostic** — switching an experiment to semi-structured pruning requires only a different experiment YAML, no code changes in any finetuning module.

```text
src/llm_pruning_mmlu/finetuning/__init__.py
src/llm_pruning_mmlu/finetuning/config.py        — FinetuningConfig, LoraConfig, WandbConfig, MaskPolicyConfig, DatasetSplitConfig
src/llm_pruning_mmlu/finetuning/mask_policy.py   — MaskEnforcer (pruning-kind-agnostic), MaskEnforcerCallback
src/llm_pruning_mmlu/finetuning/datasets.py      — MmluSftDataset with answer-token-only loss masking
src/llm_pruning_mmlu/finetuning/lora.py          — attach_lora() via PEFT
src/llm_pruning_mmlu/finetuning/wandb_utils.py   — WandbRun wrapper, no-op when disabled
src/llm_pruning_mmlu/finetuning/masked_trainer.py — train_with_masks() using HF Trainer + MaskEnforcerCallback
src/llm_pruning_mmlu/finetuning/runner.py        — run_sft_sweep() orchestrator
```

**Extensibility contract for semi-structured SFT**

`_prune_semi_structured` in `dispatch.py` now also returns its masks dict (third return value), so the runner receives `(parameters, stats, masks)` for any pruning method. `MaskEnforcer.from_prune_result` sets `pruning_kind="semi_structured"` automatically when `pruning_method` is in `SEMI_STRUCTURED_METHODS`. The `MaskEnforcerCallback` and `train_with_masks` are unchanged — they only call `apply_masks(parameters, masks)` regardless of pruning kind.

**`dispatch.py` — implemented**

`prune_model` now returns `tuple[list[PruningParameter], dict[str, Any], dict[str, torch.Tensor] | None]`. Semi-structured returns its N:M mask dict; unstructured returns `None`. Existing sweep callers updated to `targets, stats, _ = prune_model(...)`.

**`finetuning/datasets.py` — MMLU SFT format**

Each example is tokenized as `format_mmlu_prompt(question, choices) + " " + answer_letter`. Labels are set to `-100` for all prompt tokens so the cross-entropy loss is computed only on the answer token. `DataCollatorForSeq2Seq` handles padding and label padding in `masked_trainer.py`.

**`finetuning/mask_policy.py` — key abstraction**

`MaskEnforcer` holds `parameters`, `masks`, and `pruning_kind` (informational). Its `enforce()` method calls `apply_masks(parameters, masks)` — identical for structured and semi-structured. `MaskEnforcerCallback.on_step_end` fires after each optimizer step, before the next forward pass.

**`finetuning/wandb_utils.py`**

`init_wandb(cfg, run_name, config_dict)` returns a `WandbRun`. Run name format: `<model_name>_sparsity_<sparsity>_lora`. All methods (`log`, `log_summary`, `finish`) are no-ops when `enabled=False` or `WANDB_MODE=disabled`. HF Trainer's `report_to` is set via `report_to_flag(cfg)` — `"wandb"` or `"none"`.

**`finetuning/runner.py` — `run_sft_sweep`**

Uses `sft_run_id(exp_dict, ft_dict)` with prefix `structured_sft_` to avoid collision with `mmlu_pruning_` directories from prune-only sweeps.

Add a CLI entrypoint:

```text
scripts/run_sparse_finetune.py
```

The CLI should mirror the current sweep runner:

```bash
python scripts/run_sparse_finetune.py \
  --config configs/experiments/all_models_structured_sft.yaml
```

Support the same practical controls as `scripts/run_sweep.py`:

```text
--max-train-samples
--max-eval-samples
--sparsities
--fail-fast
--no-resume
```

### Slurm

Add dedicated Slurm scripts:

```text
slurm/run_all_models_structured_sft.slurm
slurm/run_gemma4_structured_sft.slurm
slurm/run_llama31_structured_sft.slurm
slurm/run_qwen3_structured_sft.slurm
```

Default all-model command:

```bash
python scripts/run_sparse_finetune.py \
  --config configs/experiments/all_models_structured_sft.yaml
```

Recommended resources:

```text
partition: gpu-h100
gres: gpu:1
cpus-per-task: 8
mem: 160G
time: 1-00:00:00
```

Expose overrides:

```text
CONFIG
MAX_TRAIN_SAMPLES
MAX_EVAL_SAMPLES
SPARSITIES
FAIL_FAST
NO_RESUME
```

---

## Output Layout

Use a new output root so sparse fine-tuning artifacts do not mix with prune-only outputs:

```text
outputs/sft_runs/
```

The run directory prefix must be `structured_sft_<config_hash>` (not `mmlu_pruning_<config_hash>`) — use the dedicated `sft_run_id_for_config` helper, not the existing `run_id_for_config` from `sweep.py`.

Recommended directory structure:

```text
outputs/sft_runs/structured_sft_<config_hash>/
  manifest.json
  combined_results.csv
  combined_results.jsonl
  llama31_8b_instruct/
    global_magnitude_structured__mlp_channel/
      sparsity_010/
        adapter/
        masks.pt
        pruning_stats.json
        training_stats.json
        metrics.json
        predictions.jsonl
        config_resolved.yaml
        run.log
      sparsity_020/
      sparsity_030/
      sparsity_040/
      sparsity_050/
  qwen3_8b/
    global_magnitude_structured__mlp_channel/
      sparsity_010/
        adapter/
        masks.pt
        pruning_stats.json
        training_stats.json
        metrics.json
        predictions.jsonl
        config_resolved.yaml
        run.log
      sparsity_020/
      sparsity_030/
      sparsity_040/
      sparsity_050/
  gemma4_e4b_it/
    global_magnitude_structured__mlp_channel/
      sparsity_010/
        adapter/
        masks.pt
        pruning_stats.json
        training_stats.json
        metrics.json
        predictions.jsonl
        config_resolved.yaml
        run.log
      sparsity_020/
      sparsity_030/
      sparsity_040/
      sparsity_050/
```

Each `metrics.json` should include:

```text
model_name
model_hf_id
pruning_method
pruning_structure
sparsity_requested
sparsity_achieved
group_sparsity_achieved
finetuning_method
train_dataset
train_split
train_samples
eval_dataset
eval_split
epochs
learning_rate
lora_r
lora_alpha
mask_policy
mmlu_accuracy
emissions_kg_co2
```

---

## Validation Plan

### Unit Tests

Add focused tests:

```text
tests/unit/test_finetuning_config.py
tests/unit/test_mask_preservation_during_training.py
tests/unit/test_sparse_finetune_resume.py
```

Minimum checks:

* Fine-tuning config parses and inherits correctly.
* Applying one optimizer step does not regrow masked base weights.
* Resume skips completed `(model, sparsity)` outputs.
* Metrics include both pruning and fine-tuning metadata.

### Smoke Test

Use a tiny run before any full job:

```bash
python scripts/run_sparse_finetune.py \
  --config configs/experiments/all_models_structured_sft.yaml \
  --max-train-samples 16 \
  --max-eval-samples 16 \
  --fail-fast
```

The smoke test passes only if:

* all five sparsity points can be restricted to one point for a quick probe, or one sparsity point can be selected with `SPARSITIES=10`;
* `training_stats.json` is written;
* `metrics.json` is written;
* masked base weights remain zero after training;
* MMLU evaluation runs after adapter loading.

### Full Run Gate

Run full all-model sparse fine-tuning only after:

* unit tests pass;
* 10% smoke runs pass for Llama, Qwen, and Gemma;
* 50% smoke runs pass for Llama, Qwen, and Gemma;
* prune-only structured baseline metrics are available for all three models.

Submit:

```bash
sbatch slurm/run_all_models_structured_sft.slurm
```

---

## Analysis Plan

Compare prune-only vs sparse fine-tuned results at each sparsity:

```text
accuracy_recovered = sft_accuracy - prune_only_accuracy
recovery_fraction = (sft_accuracy - prune_only_accuracy) / (dense_accuracy - prune_only_accuracy)
```

Generate plots:

```text
outputs/plots/structured_sft/accuracy_vs_sparsity.png
outputs/plots/structured_sft/recovery_vs_sparsity.png
outputs/plots/structured_sft/accuracy_recovered.png
outputs/plots/structured_sft/emissions_vs_recovery.png
```

Add a findings file after the first complete run:

```text
findings/FINDINGS_STRUCTURED_SFT.md
```

The findings should answer:

* Does sparse fine-tuning recover meaningful accuracy at 10-50% structured sparsity?
* Which of Llama, Qwen, and Gemma recovers best from structured sparsity?
* Which sparsity point gives the best recovery per GPU-hour?
* Does 50% structured sparsity remain usable after recovery?
* Should the next experiment raise sparsity above 50%, change MMLU split/sample strategy, or mask LoRA channels more strictly?

---

## Implementation Order

Steps 1–9 are **done**. Remaining work starts at step 10.

1. ~~Extend `dispatch.py`~~ — `prune_model` returns `(parameters, stats, masks)` for all methods.
2. ~~Add `finetuning/config.py`~~ — `FinetuningConfig`, `LoraConfig`, `WandbConfig`, `MaskPolicyConfig`, `DatasetSplitConfig`.
3. ~~Add `finetuning/wandb_utils.py`~~ — project `sparse-sft`, no-op mode.
4. ~~Add `finetuning/mask_policy.py`~~ — `MaskEnforcer` + `MaskEnforcerCallback`.
5. ~~Add `finetuning/datasets.py`~~ — `MmluSftDataset` with answer-token-only loss masking.
6. ~~Add `finetuning/lora.py`~~ and ~~`finetuning/masked_trainer.py`~~.
7. ~~Add `finetuning/runner.py`~~ with `run_sft_sweep`.
8. ~~Add `scripts/run_sparse_finetune.py`~~.
9. ~~Add all-model structured SFT configs and Slurm scripts~~.
10. Add unit tests: `test_finetuning_config.py`, `test_mask_preservation_during_training.py`, `test_sparse_finetune_resume.py`.
11. Run smoke tests at 10% and 50% for Llama, Qwen, and Gemma.
12. Run full all-model sparse fine-tuning sweep at `10, 20, 30, 40, 50`.
13. Add comparison plots and structured SFT findings.
14. Use the first full run to decide whether to test stricter LoRA masking or sparsity above 50%.

---

## Risks

### LoRA can bypass structured masks

If LoRA adapters on `gate_proj`, `up_proj`, or `down_proj` create activity in pruned channels, the experiment is no longer strict structured sparse fine-tuning. Track this explicitly with `mask_lora_pruned_channels`. Start permissive for recovery, then run a stricter follow-up.

### MMLU fine-tuning can overfit

MMLU is small relative to general instruction-tuning datasets, and repeated exposure can overfit. Keep MMLU test held out, track validation loss/accuracy, and report final test accuracy only once per completed configuration.

### Memory pressure

Gemma and 8B models may need conservative defaults:

```text
batch_size: 1
gradient_accumulation_steps: 16
bf16: true
gradient_checkpointing: true
```

### Comparability

Always evaluate with the same MMLU scoring path used by prune-only sweeps. Do not switch evaluation prompts or scoring mode during the recovery comparison.

### wandb credential availability on compute nodes

Slurm jobs on `gpu-h100` may not have `WANDB_API_KEY` in the environment. Set the key in the Slurm script or in `~/.netrc` before submitting, and verify with a `wandb login --verify` call in the smoke test. If the key is unavailable, set `finetuning.wandb.enabled: false` or pass `WANDB_MODE=offline` — the `wandb_utils.py` no-op path will keep the run from crashing.
