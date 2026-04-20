# Structured Sparsity Plan

## Project: Structured Pruning Sweep for 8B LLMs on MMLU

### Goal

Extend the current global unstructured pruning study to structured pruning while keeping the existing MMLU evaluation, artifact generation, resume behavior, emissions tracking, and plotting workflow intact.

The first useful milestone is:

> Masked global MLP-channel magnitude pruning over Llama/Qwen on MMLU, with the same artifacts as the current unstructured sweep.

This gives a clean structured-pruning baseline before adding more fragile tensor-shrinking logic.

Models:

* `meta-llama/Llama-3.1-8B-Instruct`
* `Qwen/Qwen3-8B`

Dataset:

* `cais/mmlu`

Metric:

* Accuracy

Initial structured sparsity sweep:

* `0, 10, 20, 30, 40, 50, 60, 70`

Structured pruning is more destructive per pruning decision than unstructured pruning, so the very high unstructured sparsity points (`80, 90, 95, 99`) should be deferred until recovery fine-tuning or distillation exists.

Before committing to the full sweep, run a quick sensitivity probe at `[0, 20, 40, 60]` on a 128-sample MMLU slice. If accuracy drops below 30% (near random for 4-choice) at or before 40%, cap the sweep at 50% rather than 70%. This avoids wasting GPU time on runs that produce no signal.

---

## Why this setup

The current codebase is organized around global unstructured magnitude pruning:

* Configs inherit from `configs/pruning/global_unstructured.yaml`.
* `src/llm_pruning_mmlu/experiments/sweep.py` loads each model, finds pruning parameters, computes masks, applies masks, evaluates MMLU, and writes artifacts.
* `src/llm_pruning_mmlu/pruning/magnitude.py` computes exact-count global magnitude masks over individual weights.
* `src/llm_pruning_mmlu/pruning/apply.py` applies masks in place.

The structured experiment should reuse most of that pipeline. The main change is replacing "select individual low-magnitude weights" with "select whole structured groups" such as MLP intermediate channels or attention heads.

Important distinction:

* Masked structured pruning zeros whole rows/columns but still uses dense matrix multiplications. It is useful for accuracy-vs-structure experiments.
* Materialized structured pruning physically shrinks tensors and can reduce FLOPs. It is harder and should be a second phase.

---

## High-level requirements

Implement the following:

1. Config support for structured pruning methods.
2. A pruning dispatcher so unstructured and structured methods coexist.
3. Structured group discovery for transformer MLP channels.
4. Group magnitude scoring.
5. Mask generation that zeros whole structured groups while preserving tensor shapes.
6. Structured pruning stats and artifacts.
7. Unit tests using synthetic modules.
8. A dummy structured integration config.
9. Full MMLU sweeps for Llama and Qwen after smoke tests pass.
10. Reporting that compares structured and unstructured pruning.

---

## Design decisions

### 1. Start with MLP channel pruning

The primary structured target should be MLP intermediate channels.

For Llama/Qwen-style SwiGLU MLPs, each intermediate channel must be pruned consistently across:

* `gate_proj` output row
* `up_proj` output row
* `down_proj` input column

For layer `i` and intermediate channel `j`, one structured group is:

```text
model.layers.{i}.mlp.gate_proj.weight[j, :]
model.layers.{i}.mlp.up_proj.weight[j, :]
model.layers.{i}.mlp.down_proj.weight[:, j]
```

This preserves the residual hidden size and avoids changing the transformer block interface.

### 2. Add attention-head pruning after MLP pruning works

Attention head pruning should be the secondary experiment.

**Exit criterion for MLP phase before starting this phase:** unit tests pass, the sensitivity probe (see above) completes without unexpected cliffs, and at least one full Llama sweep artifact is validated against expected accuracy degradation (accuracy at sparsity=0 matches baseline within 0.5%).

**GQA grouping definition:** Llama 3.1 8B has 32 query heads and 8 KV heads (ratio 4:1). Qwen3-8B has 64 query heads and 8 KV heads (ratio 8:1). One prunable head group is defined by one KV head index `h` and all query heads that share it:

* `q_proj` rows for each Q head in the group: indices `[h*ratio : (h+1)*ratio]` × `head_dim`
* `k_proj` row for KV head `h`: index `h` × `head_dim`
* `v_proj` row for KV head `h`: index `h` × `head_dim`
* `o_proj` input columns for each Q head in the group: indices `[h*ratio : (h+1)*ratio]` × `head_dim`

This means the prunable unit is always one KV head (with its associated Q heads), not individual Q heads. Total groups per layer = `num_key_value_heads`.

**Score formula for one KV-head group `h`:**

```text
score_h = sum over q in [h*ratio:(h+1)*ratio] of ||q_proj[q*head_dim:(q+1)*head_dim, :]||_F
        + ||k_proj[h*head_dim:(h+1)*head_dim, :]||_F
        + ||v_proj[h*head_dim:(h+1)*head_dim, :]||_F
        + sum over q in [h*ratio:(h+1)*ratio] of ||o_proj[:, q*head_dim:(q+1)*head_dim]||_F
```

This needs model config fields:

```text
hidden_size
num_attention_heads
num_key_value_heads
head_dim
```

Because Llama and Qwen can use grouped-query attention, attention pruning is more architecture-sensitive than MLP-channel pruning.

### 3. Add combined pruning last

After MLP-channel and attention-head pruning work independently, add a combined mode:

```text
structure: mlp_channel_and_attention_head
```

**Exit criterion for attention phase before starting this phase:** attention unit tests pass and at least one full Llama attention-head sweep artifact is validated.

**Scoring strategy:** MLP channels and attention head groups are scored in two separate global pools, each with its own sparsity target. The config will accept two sparsity fields:

```yaml
pruning:
  structure: mlp_channel_and_attention_head
  mlp_sparsity: 30
  head_sparsity: 20
  sparsities: [0, 10, 20, 30, 40, 50]  # applied equally to both pools
```

When `sparsities` is used (single value), the same percentage is applied independently to each pool. A joint single-pool scoring strategy (mixing channels and heads into one ranked list) is deferred — it requires a normalized score across structurally different group sizes and is harder to interpret.

This answers whether mixed structured pruning gives better accuracy retention than pruning only MLP channels or only attention heads at equal total parameter budget.

### 4. Use masks before physical tensor shrinking

The first implementation should preserve all tensor shapes and apply boolean masks.

Reasons:

* Reuses the current `apply_masks(...)` path.
* Keeps model architecture unchanged.
* Avoids fragile module replacement.
* Makes results directly comparable with the existing unstructured experiment.
* Allows correctness tests for group discovery, scoring, masking, and stats before adding tensor surgery.

Physical tensor compaction should be a second phase.

---

## Config changes

Extend `PruningConfig` in `src/llm_pruning_mmlu/config.py` with structured-specific fields:

```python
@dataclass(frozen=True)
class PruningConfig:
    method: str = "global_magnitude_unstructured"
    structure: str | None = None
    score: str = "l2_norm"
    scope: str = "global"
    target_module_types: list[str] = field(default_factory=lambda: ["Linear"])
    target_parameter_names: list[str] = field(default_factory=lambda: ["weight"])
    exclude_module_name_patterns: list[str] = field(default_factory=lambda: ["lm_head"])
    prune_bias: bool = False
    sparsities: list[float] = field(default_factory=lambda: [0, 50])
```

Add pruning config files:

```text
configs/pruning/global_structured_mlp.yaml
configs/pruning/global_structured_attention_heads.yaml
configs/pruning/global_structured_combined.yaml
```

Initial MLP config:

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
  sparsities: [0, 10, 20, 30, 40, 50, 60, 70]
```

Add experiment configs:

```text
configs/experiments/llama31_mmlu_structured_mlp_sweep.yaml
configs/experiments/qwen3_mmlu_structured_mlp_sweep.yaml
configs/experiments/all_models_mmlu_structured_mlp_sweep.yaml
configs/experiments/all_models_mmlu_structured_heads_sweep.yaml
configs/experiments/all_models_mmlu_structured_combined_sweep.yaml
```

---

## Output file structure

### Problem

The current `sparsity_dir` in `src/llm_pruning_mmlu/experiments/resume.py` generates:

```text
{run_dir}/{model_name}/sparsity_020/
```

The `run_dir` is `{output_root}/mmlu_pruning_{config_hash}`. Because structured and unstructured configs hash differently, they land in separate `run_dir`s and do not collide at the filesystem level. However:

* The path inside the run_dir carries no method or structure label — a human reading the directory cannot tell what produced the results.
* `collect_metrics` in `reporting/tables.py` globs `*/sparsity_*/metrics.json`, which is non-recursive and method-blind. If two sweeps ever share an output root without a hash difference (e.g. config key order change, float rounding), results would silently merge.
* Existing unstructured results are already on disk. Any change to the unstructured path would invalidate them.

### Solution: add a pruning tag segment for structured runs only

Extend `sparsity_dir` to accept an optional `pruning_tag`:

```python
def sparsity_dir(
    run_dir: str | Path,
    model_name: str,
    sparsity: float,
    pruning_tag: str | None = None,
) -> Path:
    base = Path(run_dir) / model_name
    if pruning_tag:
        base = base / pruning_tag
    return base / f"sparsity_{int(sparsity):03d}"
```

For unstructured runs, `pruning_tag=None` — path is identical to today (backward compatible, existing results untouched):

```text
{run_dir}/meta-llama--Llama-3.1-8B-Instruct/sparsity_020/
```

For structured runs, `pruning_tag` is `"{method}__{structure}"` (double-underscore separator):

```text
{run_dir}/meta-llama--Llama-3.1-8B-Instruct/global_magnitude_structured__mlp_channel/sparsity_020/
{run_dir}/Qwen--Qwen3-8B/global_magnitude_structured__attention_head/sparsity_030/
```

### Updated `collect_metrics` glob

Change the glob from shallow to recursive so it handles both flat and tagged layouts:

```python
# before
run_dir.glob("*/sparsity_*/metrics.json")

# after
run_dir.glob("**/sparsity_*/metrics.json")
```

This picks up both `model/sparsity_020/metrics.json` and `model/tag/sparsity_020/metrics.json` without breaking existing unstructured results.

### Full directory layout

After structured MLP sweep for Llama:

```text
{output_root}/
  mmlu_pruning_{unstructured_hash}/          # existing unstructured run — untouched
    meta-llama--Llama-3.1-8B-Instruct/
      sparsity_000/
      sparsity_020/
      ...
    Qwen--Qwen3-8B/
      sparsity_000/
      ...
    combined_results.csv
    combined_results.jsonl
    summary_by_model.csv
    manifest.json
    plots/
      sparsity_vs_accuracy.png

  mmlu_pruning_{structured_mlp_hash}/        # new structured MLP run
    meta-llama--Llama-3.1-8B-Instruct/
      global_magnitude_structured__mlp_channel/
        sparsity_000/
          config_resolved.yaml
          metrics.json
          pruning_stats.json
          emissions.json
          predictions.jsonl
          run.log
        sparsity_010/
        sparsity_020/
        ...
    Qwen--Qwen3-8B/
      global_magnitude_structured__mlp_channel/
        sparsity_000/
        ...
    combined_results.csv
    combined_results.jsonl
    summary_by_model.csv
    manifest.json
    plots/
      sparsity_vs_accuracy.png

  mmlu_pruning_{structured_heads_hash}/      # future attention-head run
    ...
```

### `should_skip` change

Pass `pruning_tag` through to `sparsity_dir`:

```python
def should_skip(
    run_dir, model_name, sparsity, resume, pruning_tag=None
) -> bool:
    if not resume:
        return False
    return metrics_complete(
        sparsity_dir(run_dir, model_name, sparsity, pruning_tag) / "metrics.json"
    )
```

### `sweep.py` change

Compute the tag from the config and pass it to both `sparsity_dir` and `should_skip`:

```python
def _pruning_tag(pruning_config) -> str | None:
    if pruning_config.method == "global_magnitude_unstructured":
        return None
    structure = pruning_config.structure or "unknown"
    return f"{pruning_config.method}__{structure}"
```

This is the only change needed in `sweep.py` beyond the dispatcher integration.

---

## Code changes

### 1. Add structured target discovery

Create:

```text
src/llm_pruning_mmlu/pruning/structured_targets.py
```

Responsibilities:

* Discover MLP channel groups.
* Later, discover attention head groups.
* Validate expected module names and tensor shapes.
* Return structured group records that point to the exact slices to score and mask.

Suggested dataclasses:

```python
@dataclass(frozen=True)
class TensorSlice:
    parameter_name: str
    parameter: torch.nn.Parameter
    dim: int
    index: int | slice


@dataclass(frozen=True)
class StructuredGroup:
    name: str
    layer_name: str
    structure: str
    slices: list[TensorSlice]
```

For MLP channel pruning, group discovery should find matching `gate_proj`, `up_proj`, and `down_proj` modules per transformer layer.

### 2. Add structured scoring and masking

Create:

```text
src/llm_pruning_mmlu/pruning/structured.py
```

Responsibilities:

* Score each structured group.
* Select groups globally by score.
* Build dense boolean masks for the affected parameters.

For MLP channel `j`, start with:

```text
score_j = ||gate_proj[j, :]||_2
        + ||up_proj[j, :]||_2
        + ||down_proj[:, j]||_2
```

Selection should use exact-count global pruning:

```text
num_groups_to_prune = floor(num_groups * sparsity / 100)
```

The mask builder should return:

```python
dict[str, torch.Tensor]
```

matching the current unstructured mask format, so `apply_masks(...)` can still be used.

### 3. Add a pruning dispatcher

Create:

```text
src/llm_pruning_mmlu/pruning/dispatch.py
```

Responsibilities:

* Keep unstructured pruning behavior unchanged.
* Route structured methods to the new structured implementation.
* Return both target parameters and pruning stats metadata.

Suggested API:

```python
def prune_model(model: torch.nn.Module, pruning_config: PruningConfig, sparsity: float) -> dict[str, Any]:
    ...
```

Behavior:

```text
method == global_magnitude_unstructured:
  find_pruning_parameters
  compute_global_magnitude_masks
  apply_masks
  return pruning_stats

method == global_magnitude_structured:
  discover structured groups
  compute structured masks
  apply_masks
  return structured pruning stats
```

Then update `src/llm_pruning_mmlu/experiments/sweep.py` to call the dispatcher instead of hardcoding `compute_global_magnitude_masks(...)`.

### 4. Extend pruning stats

Keep the existing parameter-level stats, but add structured group stats.

Example `pruning_stats.json` fields:

```json
{
  "method": "global_magnitude_structured",
  "structure": "mlp_channel",
  "score": "l2_norm",
  "scope": "global",
  "num_groups_total": 28672,
  "num_groups_pruned": 5734,
  "group_sparsity": 19.99,
  "parameter_sparsity": 20.01,
  "layers": {
    "model.layers.0.mlp": {
      "groups_total": 14336,
      "groups_pruned": 180,
      "parameter_sparsity": 1.25
    }
  }
}
```

Add these fields to `metrics.json` and combined CSV:

```text
pruning_method
pruning_structure
group_sparsity_requested
group_sparsity_achieved
parameter_sparsity_achieved
num_groups_total
num_groups_pruned
```

This matters because group sparsity and parameter sparsity may not be identical, especially once attention-head pruning is included.

---

## Testing plan

Add unit tests before launching full 8B sweeps.

New tests:

```text
tests/unit/test_structured_mlp_pruning.py
tests/unit/test_structured_attention_pruning.py
tests/unit/test_pruning_dispatch.py
tests/unit/test_structured_stats.py
```

Minimum test coverage:

* MLP group discovery finds the expected number of groups.
* `sparsity=0` leaves all weights unchanged.
* `sparsity=50` prunes exactly half of synthetic MLP groups.
* Global group pruning chooses the lowest-scoring groups across layers.
* Masks preserve original tensor shapes.
* MLP channel masks zero `gate_proj` rows, `up_proj` rows, and `down_proj` columns consistently.
* Existing unstructured tests still pass unchanged.

Add a dummy structured integration config:

```text
tests/fixtures/dummy_structured_config.yaml
```

It should run one tiny sweep at:

```text
sparsities: [0, 50]
```

---

## Experiment plan

### Stage A: local validation

```bash
python scripts/smoke_test.py
pytest
```

### Stage B: dummy structured sweep

```bash
python scripts/run_sweep.py --config tests/fixtures/dummy_structured_config.yaml
```

### Stage C: one real structured run with a small sample

```bash
python scripts/run_experiment.py \
  --config configs/experiments/llama31_mmlu_structured_mlp_sweep.yaml \
  --sparsity 20
```

If using SLURM, first run with a small sample:

```bash
sbatch --export=ALL,CONFIG=configs/experiments/llama31_mmlu_structured_mlp_sweep.yaml,MAX_SAMPLES=128 \
  slurm/run_mmlu_pruning_sweep.slurm
```

### Stage D: full MLP structured sweep

```bash
sbatch --export=ALL,CONFIG=configs/experiments/all_models_mmlu_structured_mlp_sweep.yaml \
  slurm/run_mmlu_pruning_sweep.slurm
```

### Stage E: attention-head structured sweep

Run only after MLP-channel pruning is correct.

```bash
sbatch --export=ALL,CONFIG=configs/experiments/all_models_mmlu_structured_heads_sweep.yaml \
  slurm/run_mmlu_pruning_sweep.slurm
```

### Stage F: combined structured sweep

Run only after MLP and attention-head pruning work independently.

```bash
sbatch --export=ALL,CONFIG=configs/experiments/all_models_mmlu_structured_combined_sweep.yaml \
  slurm/run_mmlu_pruning_sweep.slurm
```

---

## Emissions tracking

Emissions are already tracked via `codecarbon` through the `EmissionsTracker` context manager in `src/llm_pruning_mmlu/utils/emissions.py`. This wraps each per-model-per-sparsity run and records:

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

For structured pruning runs, ensure the following additional fields are propagated into `metrics.json` and the combined CSV alongside existing emissions fields:

```text
pruning_method        # global_magnitude_structured
pruning_structure     # mlp_channel | attention_head | mlp_channel_and_attention_head
group_sparsity_requested
group_sparsity_achieved
parameter_sparsity_achieved
```

This enables the `co2_vs_structured_sparsity.png` plot and direct comparison with the unstructured emissions baseline. No new dependency is needed — codecarbon is already an optional dependency and degrades gracefully when unavailable.

**Per-prompt emissions attribution** (already implemented in `sweep.py`) should be preserved unchanged for structured runs.

---

## Reporting plan

The structured report should compare against the existing unstructured run.

Core questions:

* At equal parameter sparsity, does structured pruning hurt accuracy more than unstructured pruning?
* Does structured pruning have an earlier or later MMLU cliff?
* Are MLP channels safer to prune than attention heads?
* Does Qwen3 remain more brittle than Llama at high sparsity?
* Does mask-based structured pruning still show flat carbon cost?
* If materialized pruning is implemented later, does latency or energy improve?

Add plots:

```text
accuracy_vs_parameter_sparsity_structured.png
accuracy_vs_group_sparsity.png
structured_vs_unstructured_accuracy.png
structured_vs_unstructured_accuracy_retained.png
co2_vs_structured_sparsity.png
```

Note: `tokens_per_second_vs_sparsity.png` is deferred. Masked structured pruning still uses dense matrix multiplications, so throughput will not improve. This plot is only meaningful after materialized (tensor-shrinking) pruning is implemented.

Most important comparison plot:

```text
structured_vs_unstructured_accuracy_retained.png
```

This directly extends the current findings from `FINDINGS.md`.

---

## Optional phase: materialized structured pruning

After masked structured pruning is correct, implement physical tensor compaction for MLP channels.

For MLP pruning, physically shrink:

```text
gate_proj.out_features
up_proj.out_features
down_proj.in_features
config.intermediate_size
```

Requirements:

* Replace affected `torch.nn.Linear` modules.
* Preserve dtype and device.
* Copy only unpruned rows/columns.
* Update model config where needed.
* Validate generation and MMLU scoring after replacement.
* Add latency and throughput measurement.

Do not start with this phase. It is the route to real FLOP reduction, but it is significantly more fragile than masked structured pruning.

---

## Recommended implementation order

1. Add structured config fields and YAML files.
2. Add MLP structured group discovery.
3. Add MLP group scoring and mask creation.
4. Add the pruning dispatcher.
5. Update `sweep.py` to use the dispatcher.
6. Extend stats, metrics, and combined CSV fields.
7. Add unit tests and dummy integration tests.
8. Run masked MLP structured sweep.
9. Add attention-head pruning.
10. Run structured-vs-unstructured reporting.
11. Consider physical MLP compaction only after masked correctness is established.

