# Semi-Structured Sparsity Plan

## Project: 2:4 and 4:8 Semi-Structured Pruning Sweep for 8B LLMs on MMLU

### Goal

Extend the current unstructured and structured pruning studies to semi-structured
N:M sparsity while keeping the same MMLU evaluation, artifact generation, resume
behavior, CodeCarbon emissions tracking, and plotting workflow.

Semi-structured sparsity is different from the current two pruning modes:

* Unstructured pruning zeros individual weights globally.
* Structured pruning zeros full MLP channels or other whole groups.
* Semi-structured pruning enforces a local hardware-oriented pattern: exactly
  `N` zeros in every contiguous block of `M` weights.

Initial semi-structured patterns:

* `2:4`: two weights kept and two weights pruned per block of four.
* `4:8`: four weights kept and four weights pruned per block of eight.

Models:

* `meta-llama/Llama-3.1-8B-Instruct`
* `Qwen/Qwen3-8B`

Dataset:

* `cais/mmlu`

Metric:

* Accuracy

Primary comparison points:

* Dense baseline: `0%`
* Semi-structured `2:4`: fixed `50%` local sparsity
* Semi-structured `4:8`: fixed `50%` local sparsity

Optional ablations after the first sweep:

* Apply the pattern only to MLP projection matrices.
* Apply the pattern to attention projection matrices.
* Compare row-wise versus column-wise block layout.
* Evaluate whether `4:8` preserves accuracy better than `2:4` at the same
  nominal sparsity because it gives the mask more freedom inside a larger block.

---

## Why this setup

The current repository already has:

* `global_magnitude_unstructured`
* `global_magnitude_structured` with `structure: mlp_channel`
* a pruning dispatcher in `src/llm_pruning_mmlu/pruning/dispatch.py`
* recursive metric collection in `src/llm_pruning_mmlu/reporting/tables.py`
* per-model-per-sparsity CodeCarbon tracking through
  `src/llm_pruning_mmlu/utils/emissions.py`

Semi-structured pruning should reuse the same dispatcher, artifacts, resume
logic, and reporting code. The main implementation difference is mask creation:
instead of selecting weights globally, create masks independently inside each
contiguous N:M block.

This first phase should use masked semi-structured pruning, not kernel-level
compressed inference. That keeps evaluation comparable to the existing masked
unstructured and masked structured experiments.

Important interpretation:

* Masked N:M pruning measures accuracy under a hardware-friendly sparsity
  pattern.
* It does not prove latency or energy savings by itself, because PyTorch dense
  matmul still executes dense kernels unless the model is converted to a runtime
  that uses semi-structured sparse kernels.
* CodeCarbon will still measure real wall-clock experiment emissions, which is
  useful for comparing experiment cost, but speed/energy gains should not be
  claimed until sparse kernels are enabled.

---

## High-level requirements

Implement the following:

1. Config support for semi-structured N:M pruning, with config-level validation
   that `nm_n` and `nm_m` are set when the method is semi-structured.
2. Dispatcher support for `global_magnitude_semi_structured`, including an
   updated error message listing all three method families.
3. N:M block mask generation for `2:4` and `4:8`.
4. Strict mask validation that verifies every complete block obeys the pattern,
   and explicitly rejects unsupported `block_dim` values.
5. Semi-structured pruning stats and artifacts.
6. Unit tests using synthetic linear layers, including edge cases for
   `in_features < M` and for `block_dim` validation.
7. Dummy integration config for a tiny sweep.
8. Full MMLU sweeps for Llama and Qwen.
9. Reporting that compares unstructured, structured, `2:4`, and `4:8`, with a
   cross-sweep CSV merge utility.
10. CodeCarbon emissions fields in `metrics.json`, `emissions.json`, and
    combined CSV output.

---

## Design decisions

### 1. Start with Linear weights only

Apply N:M masks to eligible `torch.nn.Linear.weight` tensors selected by the
existing target discovery code.

Do not prune:

* biases
* embeddings
* layer norms
* rotary embeddings
* `lm_head`

This matches the current unstructured scope and makes the comparison clean.

### 2. Use local magnitude pruning inside each block

For each contiguous block of `M` weights, keep the `M - N` largest-magnitude
weights and prune the `N` smallest-magnitude weights.

For `2:4`:

```text
block size M = 4
zeros per block N = 2
keep per block = 2
nominal sparsity = 50%
```

For `4:8`:

```text
block size M = 8
zeros per block N = 4
keep per block = 4
nominal sparsity = 50%
```

Tie handling should be deterministic. Use stable sorting or deterministic
`topk` behavior on CPU after setting seeds.

### 3. Define the first block layout clearly

Start with row-wise contiguous blocks over the input dimension of each linear
weight matrix:

```text
weight shape: [out_features, in_features]
blocks: weight[row, col:col+M]   (block_dim=1, the input/column dimension)
```

`block_dim=1` means "slide the block window along dimension 1 (columns) of the
weight matrix", which corresponds to blocking within each row. Only `block_dim=1`
is supported in phase 1. `compute_nm_magnitude_masks` and `validate_nm_mask`
must raise `ValueError` if called with any other value.

Handling non-divisible dimensions:

* Complete blocks must obey N:M exactly.
* Remainder weights at the end of a row (when `in_features % M != 0`) remain
  unpruned in phase 1.
* If `in_features < M` for a layer, every row is entirely a remainder and
  nothing is pruned for that layer. This is expected and not an error. Log a
  warning naming the layer so it is visible in the run log.
* Report both nominal sparsity and achieved sparsity because remainders make
  achieved parameter sparsity slightly below 50% when dimensions are not
  divisible by `M`.

Optional later ablation:

* column-wise blocks over the output dimension (`block_dim=0`)
* tensor-flattened blocks
* vendor-specific layout if targeting a specific sparse kernel backend

### 4. Keep 2:4 and 4:8 as two separate sweep configs

Although both patterns target about 50% sparsity, they should produce separate
run hashes and separate tagged artifact directories:

```text
global_magnitude_semi_structured__2_4/
global_magnitude_semi_structured__4_8/
```

This keeps resume behavior and reporting unambiguous.

### 5. Do not mix N:M with global sparsity percentages initially

For the first implementation, `sparsities` must be exactly:

```yaml
sparsities: [0, 50]
```

**Important semantic contract**: for `global_magnitude_semi_structured`, the
float value in `sparsities` is a trigger, not a continuous target. The N:M
pattern always produces the same mask regardless of whether the trigger value
is `50` or `70`. To prevent silent misuse, `parse_config` must raise
`ConfigError` if `method == "global_magnitude_semi_structured"` and `sparsities`
contains any value other than `0` or `50`.

At `0`, no pruning is applied. At `50`, the configured N:M pattern is applied.
The actual achieved sparsity is computed from the masks and stored in
`nm_sparsity_achieved`; it may be slightly below 50% because of remainder
weights.

Do not add arbitrary percentages such as `10, 20, 30`. N:M sparsity is defined
by the pattern, not by a free-form global sparsity target.

---

## Config changes

### PruningConfig dataclass

Extend `PruningConfig` in `src/llm_pruning_mmlu/config.py` with
semi-structured fields.  **This dataclass change must land before any
semi-structured YAML configs are written**, because `parse_config` does a
strict `PruningConfig(**data["pruning"])` unpack — unknown keys raise
`TypeError` immediately.

```python
@dataclass(frozen=True)
class PruningConfig:
    method: str = "global_magnitude_unstructured"
    structure: str | None = None
    score: str = "l2_norm"
    scope: str = "global"
    nm_n: int | None = None
    nm_m: int | None = None
    block_dim: int = 1
    target_module_types: list[str] = field(default_factory=lambda: ["Linear"])
    target_parameter_names: list[str] = field(default_factory=lambda: ["weight"])
    exclude_module_name_patterns: list[str] = field(default_factory=lambda: ["lm_head"])
    prune_bias: bool = False
    sparsities: list[float] = field(default_factory=lambda: [0, 50])
```

### Config-level validation in `parse_config`

Add the following checks to `parse_config` in `config.py` after constructing
`PruningConfig`:

```python
_SEMI_STRUCTURED_METHOD = "global_magnitude_semi_structured"

pruning_cfg = PruningConfig(**data["pruning"])

if pruning_cfg.method == _SEMI_STRUCTURED_METHOD:
    if pruning_cfg.nm_n is None or pruning_cfg.nm_m is None:
        raise ConfigError(
            "nm_n and nm_m must be set for method "
            f"{_SEMI_STRUCTURED_METHOD!r}"
        )
    allowed = {0, 50}
    bad = [s for s in pruning_cfg.sparsities if s not in allowed]
    if bad:
        raise ConfigError(
            f"For {_SEMI_STRUCTURED_METHOD!r}, sparsities must contain only "
            f"0 or 50. Got unexpected values: {bad}. The N:M pattern is "
            "determined by nm_n/nm_m, not by the sparsity float."
        )
```

### Pruning YAML configs

Use `score: l2_norm` in all new configs to stay consistent with the existing
unstructured and structured configs.  (`score` is metadata-only — no scoring
code reads this field — but inconsistent values produce confusing artifacts.)

Add:

```text
configs/pruning/global_semi_structured_2_4.yaml
configs/pruning/global_semi_structured_4_8.yaml
```

`configs/pruning/global_semi_structured_2_4.yaml`:

```yaml
pruning:
  method: global_magnitude_semi_structured
  structure: nm_2_4
  score: l2_norm
  scope: local_block
  nm_n: 2
  nm_m: 4
  block_dim: 1
  target_module_types:
    - Linear
  target_parameter_names:
    - weight
  exclude_module_name_patterns:
    - lm_head
  prune_bias: false
  sparsities: [0, 50]
```

`configs/pruning/global_semi_structured_4_8.yaml`:

```yaml
pruning:
  method: global_magnitude_semi_structured
  structure: nm_4_8
  score: l2_norm
  scope: local_block
  nm_n: 4
  nm_m: 8
  block_dim: 1
  target_module_types:
    - Linear
  target_parameter_names:
    - weight
  exclude_module_name_patterns:
    - lm_head
  prune_bias: false
  sparsities: [0, 50]
```

Add experiment configs:

```text
configs/experiments/llama31_mmlu_semi_structured_2_4_sweep.yaml
configs/experiments/qwen3_mmlu_semi_structured_2_4_sweep.yaml
configs/experiments/all_models_mmlu_semi_structured_2_4_sweep.yaml
configs/experiments/llama31_mmlu_semi_structured_4_8_sweep.yaml
configs/experiments/qwen3_mmlu_semi_structured_4_8_sweep.yaml
configs/experiments/all_models_mmlu_semi_structured_4_8_sweep.yaml
```

Example all-model config:

```yaml
inherits:
  - configs/base.yaml
  - configs/datasets/mmlu.yaml
  - configs/pruning/global_semi_structured_2_4.yaml

models:
  - configs/models/llama31_8b_instruct.yaml
  - configs/models/qwen3_8b.yaml
```

---

## Code changes

### 1. Add semi-structured mask creation

Create:

```text
src/llm_pruning_mmlu/pruning/semi_structured.py
```

Responsibilities:

* validate `0 < nm_n < nm_m`
* validate supported patterns: `(2, 4)` and `(4, 8)`
* build dense boolean masks compatible with `apply_masks(...)`
* preserve parameter shapes
* report block-level stats
* keep incomplete remainder blocks unpruned

Suggested API:

```python
def compute_nm_magnitude_masks(
    parameters: list[PruningParameter],
    n: int,
    m: int,
    block_dim: int = 1,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    ...
```

Expected output:

```python
masks = {
    "model.layers.0.mlp.gate_proj.weight": bool_tensor_same_shape,
    ...
}

stats = {
    "nm_n": 2,
    "nm_m": 4,
    "block_dim": 1,
    "num_blocks_total": 123456,
    "num_complete_blocks": 123000,
    "num_remainder_weights": 456,
    "num_weights_pruned_by_nm": 246000,
    "nm_sparsity": 50.0,
}
```

### 2. Add validation helpers

Also in `semi_structured.py`, add:

```python
def validate_nm_mask(
    parameter: torch.Tensor,
    mask: torch.Tensor,
    n: int,
    m: int,
    block_dim: int = 1,
) -> dict[str, Any]:
    ...
```

This should verify:

* `block_dim == 1`; raise `ValueError` immediately if any other value is passed
  (only row-wise blocks are supported in phase 1)
* every complete block has exactly `n` zeros
* every complete block has exactly `m - n` kept weights
* remainder weights are kept (all True in the mask)
* mask shape matches parameter shape

### 3. Extend the pruning dispatcher

Update `src/llm_pruning_mmlu/pruning/dispatch.py`:

```python
_SEMI_STRUCTURED_METHODS = {"global_magnitude_semi_structured"}
```

Dispatch behavior:

```text
method == global_magnitude_unstructured:
  existing path

method == global_magnitude_structured:
  existing structured path

method == global_magnitude_semi_structured:
  if sparsity == 0:
    do not apply masks
  else:
    compute N:M masks  (pass nm_n, nm_m, block_dim from config; do NOT pass
                        the sparsity float — it is a trigger, not a target)
    validate masks
    apply masks
  return pruning_stats plus N:M stats
```

Also update the fallback `ValueError` in `prune_model` so the error message
includes all three method sets:

```python
raise ValueError(
    f"Unknown pruning method: {method!r}. "
    f"Supported: {sorted(_UNSTRUCTURED_METHODS | _STRUCTURED_METHODS | _SEMI_STRUCTURED_METHODS)}"
)
```

### 4. Update pruning tags

Update `_pruning_tag(...)` in `src/llm_pruning_mmlu/experiments/sweep.py` so
semi-structured runs get a readable tag:

```python
def _pruning_tag(pruning_config: PruningConfig) -> str | None:
    if pruning_config.method == "global_magnitude_unstructured":
        return None
    if pruning_config.method == "global_magnitude_semi_structured":
        n = pruning_config.nm_n
        m = pruning_config.nm_m
        return f"{pruning_config.method}__{n}_{m}"
    structure = pruning_config.structure or "unknown"
    return f"{pruning_config.method}__{structure}"
```

### 5. Extend metrics fields

Add these fields to `metrics.json`, `pruning_stats.json`, and combined CSV when
the method is semi-structured:

```text
pruning_method
pruning_structure
nm_n
nm_m
nm_pattern
nm_block_dim
nm_sparsity_requested
nm_sparsity_achieved
num_nm_blocks_total
num_nm_complete_blocks
num_nm_remainder_weights
num_nm_pattern_violations
parameter_sparsity_achieved
```

**`sparsity_requested` clarification**: the existing `sparsity_requested` field
in `metrics.json` will hold `50` for a semi-structured run, which is the
trigger value, not a free-form target. Add `nm_sparsity_requested` (always 50
for both N:M patterns) and `nm_sparsity_achieved` (computed from masks,
slightly below 50 when remainders exist) so consumers can use unambiguous
fields. Document in the run log that for semi-structured runs `sparsity_requested`
is a trigger.

Use `parameter_sparsity_achieved` as the cross-method x-axis when comparing
unstructured, structured, and semi-structured pruning.

### 6. Update sweep.py pruning log line

The existing log line in `sweep.py` prints `num_groups_total` and
`num_groups_pruned`, which will show `n/a` for semi-structured runs and is
misleading. Extend it:

```python
_log.info(
    "Pruning done: method=%s structure=%s achieved_sparsity=%.4f%% "
    "total_params=%d nonzero=%d groups_total=%s groups_pruned=%s "
    "nm_n=%s nm_m=%s nm_sparsity=%.2f%%",
    stats.get("method"),
    stats.get("structure"),
    stats["sparsity"],
    stats["total"],
    stats["nonzero"],
    stats.get("num_groups_total", "n/a"),
    stats.get("num_groups_pruned", "n/a"),
    stats.get("nm_n", "n/a"),
    stats.get("nm_m", "n/a"),
    stats.get("nm_sparsity", 0.0),
)
```

### 7. Add cross-sweep CSV merge utility

Each sweep writes results to its own `run_dir` (keyed by config hash), so
`combined_results.csv` within a single run contains only one pruning method.
The comparison plots (`method_accuracy_at_50_percent.png` etc.) require data
from unstructured, structured, and semi-structured runs on the same axes.

Add a utility script:

```text
scripts/merge_sweep_results.py
```

Usage:

```bash
python scripts/merge_sweep_results.py \
  --runs outputs/mmlu_pruning_<hash_unstructured> \
         outputs/mmlu_pruning_<hash_structured> \
         outputs/mmlu_pruning_<hash_2_4> \
         outputs/mmlu_pruning_<hash_4_8> \
  --out outputs/merged_comparison/
```

The script calls `collect_metrics` on each run directory, concatenates the
rows, writes `merged_combined_results.csv`, and calls the existing comparison
plot functions. This is the prerequisite for all cross-method plots in the
reporting plan.

---

## Output file structure

The existing structured tag layout should be reused.

For `2:4`:

```text
{output_root}/mmlu_pruning_{hash}/
  llama31_8b_instruct/
    global_magnitude_semi_structured__2_4/
      sparsity_000/
        config_resolved.yaml
        metrics.json
        pruning_stats.json
        emissions.json
        predictions.jsonl
        run.log
      sparsity_050/
        ...
  qwen3_8b/
    global_magnitude_semi_structured__2_4/
      sparsity_000/
      sparsity_050/
  combined_results.csv
  combined_results.jsonl
  summary_by_model.csv
  manifest.json
  plots/
    sparsity_vs_accuracy.png
```

For `4:8`:

```text
{output_root}/mmlu_pruning_{hash}/
  llama31_8b_instruct/
    global_magnitude_semi_structured__4_8/
      sparsity_000/
      sparsity_050/
  qwen3_8b/
    global_magnitude_semi_structured__4_8/
      sparsity_000/
      sparsity_050/
```

---

## Testing plan

### New files

Create two new unit test files:

```text
tests/unit/test_semi_structured_pruning.py
tests/unit/test_semi_structured_stats.py
```

### Extend the existing dispatch test file

`tests/unit/test_pruning_dispatch.py` already exists as part of the structured
sweep work (it is untracked in git). Do **not** recreate it. Add semi-structured
dispatch cases to the existing file.

### Minimum test coverage

**Mask correctness:**

* `2:4` masks prune exactly two weights in every complete block of four.
* `4:8` masks prune exactly four weights in every complete block of eight.
* The lowest-magnitude weights in each block are pruned (not highest).
* `sparsity=0` leaves all weights unchanged.
* Remainder weights are not pruned (all True in mask).
* Masks preserve original tensor shapes.

**Edge cases (must be covered):**

* `in_features < M`: e.g., a `Linear(2, 4)` layer with `M=4`. Every row is a
  remainder; no weights are pruned and the function must not raise. Check that
  `num_complete_blocks == 0` and `nm_sparsity == 0.0` in stats.
* Non-divisible `in_features`: e.g., `in_features=6` with `M=4`. First block
  (cols 0–3) is pruned; last two weights (cols 4–5) are remainder and kept.
* `block_dim != 1` passed to `compute_nm_magnitude_masks` or
  `validate_nm_mask`: must raise `ValueError`.

**Validation:**

* Invalid patterns (e.g., `n=0`, `n >= m`) fail clearly in `validate_nm_mask`.
* `validate_nm_mask` raises on shape mismatch between parameter and mask.

**Dispatcher (add to `test_pruning_dispatch.py`):**

* Dispatcher routes `global_magnitude_semi_structured` correctly.
* Semi-structured stats keys (`nm_n`, `nm_m`, `nm_sparsity`, etc.) are present.
* `sparsity=0` with semi-structured config leaves all weights unchanged.
* Existing unstructured and structured tests still pass (no regression).

Add dummy integration configs:

```text
tests/fixtures/dummy_semi_structured_2_4_config.yaml
tests/fixtures/dummy_semi_structured_4_8_config.yaml
```

Each should run:

```yaml
pruning:
  sparsities: [0, 50]
```

---

## Experiment plan

### Stage A: local validation

```bash
python scripts/smoke_test.py
pytest
```

### Stage B: dummy semi-structured sweeps

```bash
python scripts/run_sweep.py --config tests/fixtures/dummy_semi_structured_2_4_config.yaml
python scripts/run_sweep.py --config tests/fixtures/dummy_semi_structured_4_8_config.yaml
```

### Stage C: one-model 128-sample sensitivity probe

Run Llama first:

```bash
python scripts/run_sweep.py \
  --config configs/experiments/llama31_mmlu_semi_structured_2_4_sweep.yaml \
  --max-samples 128

python scripts/run_sweep.py \
  --config configs/experiments/llama31_mmlu_semi_structured_4_8_sweep.yaml \
  --max-samples 128
```

If using SLURM:

```bash
sbatch --export=ALL,CONFIG=configs/experiments/llama31_mmlu_semi_structured_2_4_sweep.yaml,MAX_SAMPLES=128 \
  slurm/run_mmlu_pruning_sweep.slurm

sbatch --export=ALL,CONFIG=configs/experiments/llama31_mmlu_semi_structured_4_8_sweep.yaml,MAX_SAMPLES=128 \
  slurm/run_mmlu_pruning_sweep.slurm
```

### Stage D: full 2:4 sweep

```bash
sbatch --export=ALL,CONFIG=configs/experiments/all_models_mmlu_semi_structured_2_4_sweep.yaml \
  slurm/run_mmlu_pruning_sweep.slurm
```

### Stage E: full 4:8 sweep

```bash
sbatch --export=ALL,CONFIG=configs/experiments/all_models_mmlu_semi_structured_4_8_sweep.yaml \
  slurm/run_mmlu_pruning_sweep.slurm
```

### Stage F: optional target-scope ablations

After the full all-linear sweep is validated, add configs for:

```text
semi_structured_2_4_mlp_only
semi_structured_2_4_attention_only
semi_structured_4_8_mlp_only
semi_structured_4_8_attention_only
```

These require target-name filtering such as:

```yaml
include_module_name_patterns:
  - mlp
```

or:

```yaml
include_module_name_patterns:
  - self_attn
```

Add include filters only if needed for these ablations. Do not complicate the
first implementation.

---

## CodeCarbon emissions tracking

CodeCarbon is already included in `requirements.txt` and wrapped by
`EmissionsTracker` in `src/llm_pruning_mmlu/utils/emissions.py`.

Keep the existing per-model-per-sparsity tracking behavior:

```text
load model -> apply pruning -> evaluate MMLU
```

inside one CodeCarbon tracker context.

For every semi-structured run, save:

```text
emissions_kg_co2
energy_consumed_kwh
duration_s
cpu_power_w
gpu_power_w
ram_power_w
country_name
cloud_provider
codecarbon_version
```

Also propagate emissions into:

* `emissions.json`
* `metrics.json`
* `combined_results.csv`
* per-prompt `predictions.jsonl` attribution, preserving the current logic in
  `sweep.py`

Add these semi-structured fields beside the emissions fields:

```text
pruning_method
pruning_structure
nm_pattern
nm_sparsity_requested
nm_sparsity_achieved
parameter_sparsity_achieved
```

This enables direct plots:

```text
accuracy_vs_co2_by_method.png
co2_vs_pruning_method.png
co2_vs_parameter_sparsity.png
carbon_efficiency_by_method.png
```

Interpretation guardrail:

* If masked semi-structured pruning uses dense kernels, CodeCarbon may show
  similar or slightly higher energy than dense baseline because pruning adds
  overhead. Do not present this as evidence that N:M sparsity lacks runtime
  value. It only means this implementation has not enabled sparse kernels.

---

## Reporting plan

The report should compare:

* dense baseline
* global unstructured at 50%
* structured MLP-channel at nearest comparable parameter sparsity
* semi-structured `2:4`
* semi-structured `4:8`

Core questions:

* At roughly 50% parameter sparsity, does `2:4` preserve more MMLU accuracy
  than global unstructured pruning?
* Does `4:8` preserve more accuracy than `2:4` at the same nominal sparsity?
* Is Llama or Qwen more robust under N:M constraints?
* Are emissions dominated by evaluation time rather than pruning method?
* Does masked N:M pruning show any measurable wall-clock or energy change
  before sparse kernels are introduced?

Add plots:

```text
semi_structured_accuracy_comparison.png
semi_structured_accuracy_retained.png
semi_structured_2_4_vs_4_8.png
method_accuracy_at_50_percent.png
method_co2_at_50_percent.png
accuracy_vs_co2_by_method.png
```

Most important comparison plot:

```text
method_accuracy_at_50_percent.png
```

This directly compares the practical 50% sparsity point across pruning methods.

---

## Optional phase: real semi-structured sparse kernels

After masked N:M correctness is established, evaluate whether the models can use
actual semi-structured sparse inference.

Possible directions:

* `torch.sparse` semi-structured tensor support if compatible with the target
  GPU, dtype, and matrix shapes.
* NVIDIA 2:4 sparse Tensor Core path on supported GPUs.
* Third-party inference runtime with N:M support.

Requirements before claiming speed or carbon improvements:

* Verify sparse kernel execution, not just masked dense weights.
* Add latency and throughput metrics:

```text
tokens_per_second
samples_per_second
eval_duration_s
peak_gpu_memory_gb
```

* Compare dense, masked N:M, and kernel-enabled N:M under identical evaluation
  settings.
* Keep CodeCarbon enabled for all three modes.

This should be a second phase. The first phase is an accuracy and emissions
accounting baseline under correct N:M masks.

---

## Recommended implementation order

**The dataclass must come first.** `parse_config` does a strict keyword-unpack
of `PruningConfig(**data["pruning"])`; any YAML written before the new fields
exist will raise `TypeError` at load time.

1. Add `nm_n`, `nm_m`, `block_dim` fields to `PruningConfig` in `config.py`.
2. Add config-level validation to `parse_config` (semi-structured method requires
   `nm_n`/`nm_m`; `sparsities` must be only `[0, 50]`).
3. Add `global_semi_structured_2_4.yaml` and `global_semi_structured_4_8.yaml`
   (use `score: l2_norm` for consistency with existing configs).
4. Add experiment configs for Llama, Qwen, and both models.
5. Implement `pruning/semi_structured.py` (mask generation, `block_dim=1`-only
   enforcement, remainder-skip logic with a warning for `in_features < M`,
   `validate_nm_mask`).
6. Extend the dispatcher: add `_SEMI_STRUCTURED_METHODS`, add `_prune_semi_structured`,
   update the fallback `ValueError` to include all three method sets.
7. Extend `_pruning_tag` in `sweep.py` for `global_magnitude_semi_structured`.
8. Extend stats and metrics fields; update the pruning log line in `sweep.py`.
9. Add `scripts/merge_sweep_results.py` for cross-sweep CSV aggregation.
10. Add `tests/unit/test_semi_structured_pruning.py` and
    `tests/unit/test_semi_structured_stats.py` (new files).
11. Extend `tests/unit/test_pruning_dispatch.py` (existing file — do not recreate)
    with semi-structured dispatch cases.
12. Add dummy integration configs.
13. Run smoke tests and `pytest`.
14. Run 128-sample probes.
15. Submit full 2:4 and 4:8 SLURM sweeps.
16. Run `scripts/merge_sweep_results.py` to produce cross-method comparison CSV,
    then generate comparison plots.
17. Consider sparse-kernel inference only after masked N:M results are correct.

