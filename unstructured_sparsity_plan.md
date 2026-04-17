# PLAN.md

## Project: Unstructured Pruning Sweep for 8B LLMs on MMLU

### Goal

Build a reproducible evaluation pipeline that applies **global unstructured pruning** to two 8B LLMs across a fixed sparsity sweep and evaluates performance on **MMLU** using **accuracy**.

Models:

* `meta-llama/Llama-3.1-8B-Instruct`
* `Qwen/Qwen3-8B`

Dataset:

* `cais/mmlu`

Metric:

* Accuracy

Sparsity sweep:

* `0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99`

This plan is designed so Codex can implement the full pipeline with a robust file structure, configuration system, tests, logging, artifacts, and resumable experiment runs.

---

## Why this setup

* `meta-llama/Llama-3.1-8B-Instruct` is the official Hugging Face Llama 3.1 8B instruction-tuned model.
* `Qwen/Qwen3-8B` is the current dense Qwen3 8B model on Hugging Face.
* `cais/mmlu` is a standard multiple-choice benchmark with a simple and stable **accuracy** metric.

Notes:

* Unstructured pruning is best treated here as an **accuracy-vs-sparsity** study, not as a latency benchmark.
* Real inference speedups from irregular sparsity typically require sparse kernels and hardware-specific runtime support.

---

## High-level requirements

Codex should implement the following:

1. A clean Python package for pruning and evaluation.
2. YAML-driven experiment configs.
3. Support for the two requested models.
4. Global unstructured pruning over target linear weights.
5. MMLU evaluation with deterministic prompt formatting.
6. Saving of predictions, metrics, configs, logs, and plots.
7. Resume support so interrupted runs continue safely.
8. Unit tests for pruning, metrics, config loading, and parsing.
9. Small smoke tests that run without 8B models.
10. CI-ready structure.

---

## Design decisions

### 1. Evaluation framing

Use **zero-shot multiple-choice evaluation** on MMLU.

For each example:

* Format a prompt with question and answer choices A/B/C/D.
* Score each candidate answer by token log-probability or by single-token label generation, depending on tokenizer behavior.
* Choose the highest-scoring option.
* Compare with gold label.

Preferred implementation:

* Use **choice scoring via conditional log-likelihood** rather than open-ended generation.
* This is more stable for benchmarking and easier to compare across pruning levels.

### 2. Pruning scope

Apply pruning to:

* All `torch.nn.Linear` weights inside the transformer blocks

Do not prune:

* biases
* embeddings initially
* layer norms
* rotary embeddings or other non-linear modules
* lm_head by default

Reason:

* This is a practical and standard first setup.
* It reduces implementation risk and keeps the experiment interpretable.

Optional later extension:

* Add config flags for pruning embeddings and lm_head.

### 3. Pruning method

Use:

* **global unstructured magnitude pruning**

Meaning:

* Collect all target weights across eligible modules
* Compute a global magnitude threshold
* Zero out the smallest weights until desired sparsity is reached

Why:

* Global magnitude pruning is the most common strong baseline.
* It is simple, reproducible, and easy to test.

### 4. Precision and loading

Support:

* `bfloat16` if available
* otherwise `float16`
* CPU fallback for tests only

For real runs:

* use `device_map="auto"` or explicit single-GPU placement
* support 4-bit loading only for development experiments, not for the main pruning study

Important:

* Do **not** combine the main pruning benchmark with quantized weights unless explicitly configured.
* Keep the main study in standard dense precision so pruning effects are isolated.

### 5. Reproducibility

Every run must save:

* exact model id
* dataset split details
* pruning configuration
* seed
* torch / transformers / datasets versions
* hardware info if available
* timestamp
* git commit if repository is in git

---

## Proposed repository structure

```text
llm-pruning-mmlu/
├── README.md
├── PLAN.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── .env.example
├── Makefile
├── configs/
│   ├── base.yaml
│   ├── models/
│   │   ├── llama31_8b_instruct.yaml
│   │   └── qwen3_8b.yaml
│   ├── datasets/
│   │   └── mmlu.yaml
│   ├── pruning/
│   │   └── global_unstructured.yaml
│   └── experiments/
│       ├── llama31_mmlu_sweep.yaml
│       ├── qwen3_mmlu_sweep.yaml
│       └── all_models_mmlu_sweep.yaml
├── scripts/
│   ├── run_experiment.py
│   ├── run_sweep.py
│   ├── evaluate_model.py
│   ├── export_summary.py
│   ├── plot_results.py
│   ├── validate_env.py
│   └── smoke_test.py
├── src/
│   └── llm_pruning_mmlu/
│       ├── __init__.py
│       ├── config.py
│       ├── registry.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── io.py
│       │   ├── logging_utils.py
│       │   ├── seed.py
│       │   ├── device.py
│       │   ├── versioning.py
│       │   └── hashing.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── mmlu.py
│       │   ├── prompting.py
│       │   └── collate.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   ├── tokenizer.py
│       │   └── model_info.py
│       ├── pruning/
│       │   ├── __init__.py
│       │   ├── targets.py
│       │   ├── magnitude.py
│       │   ├── masks.py
│       │   ├── apply.py
│       │   └── stats.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── scorer.py
│       │   ├── metrics.py
│       │   ├── runner.py
│       │   └── predictions.py
│       ├── experiments/
│       │   ├── __init__.py
│       │   ├── sweep.py
│       │   ├── resume.py
│       │   └── artifacts.py
│       └── reporting/
│           ├── __init__.py
│           ├── tables.py
│           └── plots.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_pruning_targets.py
│   │   ├── test_magnitude_pruning.py
│   │   ├── test_pruning_stats.py
│   │   ├── test_prompt_formatting.py
│   │   ├── test_metrics.py
│   │   ├── test_resume.py
│   │   └── test_io.py
│   ├── integration/
│   │   ├── test_smoke_eval_tiny_model.py
│   │   ├── test_end_to_end_dummy_sweep.py
│   │   └── test_artifact_generation.py
│   └── fixtures/
│       ├── tiny_mmlu.jsonl
│       └── dummy_config.yaml
├── outputs/
│   ├── runs/
│   ├── summaries/
│   └── plots/
└── notebooks/
    └── inspect_results.ipynb
```

---

## File-by-file responsibilities

### `README.md`

Include:

* project overview
* environment setup
* how to access Llama model license if needed
* how to run one evaluation
* how to run full sweep
* how to run tests
* expected outputs

### `pyproject.toml`

Use modern packaging and tool config.
Recommended tools:

* `setuptools` or `hatchling`
* `pytest`
* `ruff`
* `black`
* optional `mypy`

### `requirements.txt`

Core dependencies:

* `torch`
* `transformers`
* `datasets`
* `accelerate`
* `pyyaml`
* `pandas`
* `numpy`
* `matplotlib`
* `tqdm`

Optional:

* `huggingface_hub`
* `safetensors`

### `requirements-dev.txt`

Add:

* `pytest`
* `pytest-cov`
* `ruff`
* `black`
* `mypy`

---

## Configuration system

Use YAML configs with dataclass-backed validation.

Config merge rule: when using `inherits`, later entries override earlier ones for
scalar values; lists are replaced (not appended). Implement this explicitly in
`config.py` — undefined merge behavior will produce silent wrong configs.

### `configs/base.yaml`

Should define shared defaults:

```yaml
seed: 42
output_root: outputs/runs
log_level: INFO
save_predictions: true
save_pruned_checkpoint: false
resume: true

device:
  dtype: bfloat16
  device_map: auto
  trust_remote_code: false

evaluation:
  batch_size: 4
  max_samples: null
  subjects: all
  split: test
  few_shot: 0
  scoring_mode: choice_logprob

reporting:
  save_csv: true
  save_json: true
  save_plot: true
```

### `configs/models/llama31_8b_instruct.yaml`

```yaml
model:
  name: llama31_8b_instruct
  hf_id: meta-llama/Llama-3.1-8B-Instruct
  family: llama
  chat_template: true
```

### `configs/models/qwen3_8b.yaml`

```yaml
model:
  name: qwen3_8b
  hf_id: Qwen/Qwen3-8B
  family: qwen
  chat_template: true
  generation_kwargs:
    enable_thinking: false   # Qwen3 activates a thinking/reasoning mode by default;
                             # disable it to prevent <think> tokens corrupting output
                             # if the fallback generation path is ever triggered
```

### `configs/datasets/mmlu.yaml`

```yaml
dataset:
  name: mmlu
  hf_id: cais/mmlu
  split: test
  subjects: all
  answer_choices: [A, B, C, D]
```

### `configs/pruning/global_unstructured.yaml`

```yaml
pruning:
  method: global_magnitude_unstructured
  target_module_types:
    - Linear
  target_parameter_names:
    - weight
  exclude_module_name_patterns:
    - lm_head
  prune_bias: false
  sparsities: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
```

### `configs/experiments/all_models_mmlu_sweep.yaml`

```yaml
inherits:
  - configs/base.yaml
  - configs/datasets/mmlu.yaml
  - configs/pruning/global_unstructured.yaml

models:
  - configs/models/llama31_8b_instruct.yaml
  - configs/models/qwen3_8b.yaml
```

---

## Core implementation details

## 1. Model loading

Implement in `src/llm_pruning_mmlu/models/loader.py`.

Responsibilities:

* load tokenizer
* load model with `AutoModelForCausalLM`
* set eval mode
* move to device
* return both objects

Implementation notes:

* prefer `torch_dtype=torch.bfloat16` when supported
* support license-gated access for Llama via environment token
* optionally set `attn_implementation` only if safe and supported
* do not mutate model before baseline evaluation except to ensure eval mode

Function sketch:

```python
def load_model_and_tokenizer(model_cfg, device_cfg):
    ...
```

## 2. Pruning target discovery

Implement in `pruning/targets.py`.

Responsibilities:

* walk model modules
* identify eligible `nn.Linear` layers
* return parameter references for pruning
* support exclusions by module name pattern

Need tests for:

* finding only linear weights
* excluding `lm_head`
* not returning biases

## 3. Global magnitude pruning

Implement in `pruning/magnitude.py` and `pruning/apply.py`.

Responsibilities:

* flatten absolute values of all target weights
* compute global threshold for requested sparsity
* create boolean masks
* apply masks in-place by zeroing weights
* store pruning metadata

Important:

* For `0%`, masks should be all ones.
* For `99%`, ensure exact or very near exact sparsity given tensor discreteness.
* Keep the implementation framework-independent instead of relying completely on `torch.nn.utils.prune`, because explicit masks are easier to inspect and test.
* **`device_map="auto"` correctness**: when the model is sharded across devices,
  weight tensors live on different CUDA devices. The global threshold computation
  must gather all target weights to a common device (e.g. CPU) before computing
  the threshold, then push masks back to each weight's original device before
  applying them in-place. Failing to do this will produce incorrect masks or
  device mismatch errors.

Suggested API:

```python
def compute_global_magnitude_masks(parameters, sparsity: float) -> dict[str, torch.Tensor]:
    ...

def apply_masks(parameters, masks) -> None:
    ...
```

## 4. Sparsity statistics

Implement in `pruning/stats.py`.

Track:

* total parameters in target set
* nonzero parameters in target set
* achieved sparsity
* optional model-wide sparsity
* per-layer sparsity summary

Save to JSON after each run.

## 5. MMLU dataset loader

Implement in `data/mmlu.py`.

Responsibilities:

* load `cais/mmlu`
* merge or iterate through all subjects
* normalize labels to A/B/C/D
* support optional subsetting by subject
* support optional `max_samples`

Need deterministic ordering.

## 6. Prompt formatting

Implement in `data/prompting.py`.

Use a stable plain format like:

```text
Question: {question}
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}
Answer:
```

Keep it simple.
Do not rely on chat generation format for the core benchmark unless absolutely necessary.

Reason:

* multiple-choice log-prob scoring is cleaner with plain prompts
* reduces template-related variance across model families

## 7. Choice scoring

Implement in `evaluation/scorer.py`.

Preferred method:

* For each prompt, append one candidate label: `A`, `B`, `C`, or `D`
* Compute conditional log-probability for each candidate answer
* Select argmax

Need to handle tokenizer edge cases:

* some tokenizers may tokenize `A` differently from ` A`
* define a canonical answer tokenization scheme
* include tests to ensure labels are comparable

Fallback method:

* constrained generation of one token followed by parsing
* only use if log-prob route is not supported cleanly

## 8. Metrics

Implement in `evaluation/metrics.py`.

Minimum metric:

* accuracy

Optional extras to save:

* per-subject accuracy
* total correct
* total samples

## 9. Experiment runner

Implement in `evaluation/runner.py` and `experiments/sweep.py`.

Flow:

1. Load config
2. Load model and tokenizer
3. Evaluate baseline at 0 if requested
4. For each sparsity:

   * reload fresh model or clone clean weights
   * apply pruning
   * evaluate on MMLU
   * save artifacts
5. Aggregate results
6. Export summary CSV and plots

Important decision:

* **Always reload a fresh model per sparsity level** rather than pruning cumulatively.

Reason:

* Cumulative pruning can introduce path dependence.
* Independent pruning per sparsity is the cleaner design.

## 10. Resume support

Implement in `experiments/resume.py`.

Behavior:

* each run should have a stable run id derived from config hash
* if result artifact for `(model, sparsity)` already exists and is valid, skip it
* if a run crashes midway, rerun only missing points

Validation:

* result is considered complete only if metrics JSON exists and passes schema checks

## 11. Reporting

Implement in `reporting/tables.py` and `reporting/plots.py`.

Artifacts to generate:

* `results.csv`
* `results.jsonl`
* `summary_by_model.csv`
* `sparsity_vs_accuracy.png`
* optional per-subject CSVs

Suggested plot:

* x-axis: sparsity
* y-axis: accuracy
* one line per model

---

## Output structure

Each completed run should create something like:

```text
outputs/runs/
└── 2026-04-16_all_models_mmlu/
    ├── manifest.json
    ├── combined_results.csv
    ├── combined_results.jsonl
    ├── plots/
    │   └── sparsity_vs_accuracy.png
    ├── llama31_8b_instruct/
    │   ├── sparsity_000/
    │   │   ├── config_resolved.yaml
    │   │   ├── metrics.json
    │   │   ├── pruning_stats.json
    │   │   ├── predictions.jsonl
    │   │   └── run.log
    │   ├── sparsity_010/
    │   ├── ...
    │   └── sparsity_099/
    └── qwen3_8b/
        ├── sparsity_000/
        ├── ...
        └── sparsity_099/
```

### Example `metrics.json`

```json
{
  "model_name": "llama31_8b_instruct",
  "model_hf_id": "meta-llama/Llama-3.1-8B-Instruct",
  "dataset": "cais/mmlu",
  "split": "test",
  "metric": "accuracy",
  "sparsity_requested": 90,
  "sparsity_achieved": 90.0004,
  "num_total_target_params": 6815744000,
  "num_nonzero_target_params": 681574400,
  "accuracy": 0.4123,
  "num_samples": 14042,
  "seed": 42,
  "timestamp": "2026-04-16T18:40:00Z"
}
```

### Example `predictions.jsonl`

```jsonl
{"subject":"abstract_algebra","question_id":"...","gold":"B","pred":"B","correct":1}
{"subject":"anatomy","question_id":"...","gold":"D","pred":"A","correct":0}
```

---

## Scripts to implement

## `scripts/validate_env.py`

Checks:

* Python version
* torch available
* CUDA availability
* transformers version
* HF token presence if needed
* access to model ids if possible

## `scripts/evaluate_model.py`

Use case:

* run one model at one sparsity level

Example:

```bash
python scripts/evaluate_model.py \
  --config configs/experiments/llama31_mmlu_sweep.yaml \
  --model-config configs/models/llama31_8b_instruct.yaml \
  --sparsity 50
```

## `scripts/run_sweep.py`

Use case:

* run all configured models and all configured sparsity levels

Example:

```bash
python scripts/run_sweep.py \
  --config configs/experiments/all_models_mmlu_sweep.yaml
```

Features:

* prints progress table
* supports `--resume`
* supports `--max-samples` override for debugging
* supports `--subjects` subset override

## `scripts/export_summary.py`

Collect all run artifacts into combined CSV and JSONL.

## `scripts/plot_results.py`

Read summary CSV and create line plot.

## `scripts/smoke_test.py`

Run a tiny end-to-end test with:

* a tiny local model or small public model
* a tiny fixture dataset
* sparsities `[0, 50]`

This should finish quickly and be used in CI.

---

## Testing plan

Codex should implement tests early, not at the end.

## Unit tests

### `test_config.py`

Validate:

* config inheritance and merge behavior
* required fields exist
* bad configs fail clearly

### `test_pruning_targets.py`

Validate:

* only `Linear.weight` parameters are selected
* exclusions work
* bias is excluded

### `test_magnitude_pruning.py`

Validate:

* 0% pruning keeps all weights
* 50% pruning zeros about half the target weights
* 99% pruning zeros nearly all target weights
* thresholding behaves consistently on known tensors

### `test_pruning_stats.py`

Validate:

* counted zeros/nonzeros are correct
* achieved sparsity is computed correctly

### `test_prompt_formatting.py`

Validate:

* prompt contains question and all four choices in fixed order
* answer suffix formatting is deterministic

### `test_metrics.py`

Validate:

* accuracy computation matches expected values

### `test_resume.py`

Validate:

* completed runs are skipped
* incomplete runs are rerun

### `test_io.py`

Validate:

* JSON and CSV artifact writing works
* directories are created safely

## Integration tests

### `test_smoke_eval_tiny_model.py`

Use a tiny causal LM and fixture dataset to verify:

* model loads
* evaluation runs
* predictions save

### `test_end_to_end_dummy_sweep.py`

Run a sweep with:

* tiny model
* tiny dataset
* sparsities `[0, 50, 90]`

Validate:

* result folders exist
* summary CSV is generated
* plots are generated

### `test_artifact_generation.py`

Validate the exact artifact schema and required files.

---

## Recommended development order for Codex

### Phase 1: Skeleton

1. Create package structure
2. Add config system
3. Add logging and artifact helpers
4. Add tests for config and IO

### Phase 2: Core pruning

1. Implement target discovery
2. Implement global magnitude mask computation
3. Implement mask application
4. Implement sparsity stats
5. Add pruning unit tests

### Phase 3: MMLU evaluation

1. Implement dataset loader
2. Implement prompt formatter
3. Implement choice scoring
4. Implement accuracy metric
5. Add evaluation tests with tiny fixture data

### Phase 4: Experiment orchestration

1. Implement single-run evaluation script
2. Implement sweep runner
3. Implement resume behavior
4. Implement summary export and plotting
5. Add integration tests

### Phase 5: Polishing

1. Improve CLI ergonomics
2. Add README examples
3. Add smoke test script
4. Add optional notebook for result inspection

---

## CLI examples Codex should support

### Validate environment

```bash
python scripts/validate_env.py
```

### Run one baseline model

```bash
python scripts/evaluate_model.py \
  --config configs/experiments/llama31_mmlu_sweep.yaml \
  --sparsity 0
```

### Run one sparsity point

```bash
python scripts/evaluate_model.py \
  --config configs/experiments/qwen3_mmlu_sweep.yaml \
  --sparsity 90
```

### Run complete sweep for one model

```bash
python scripts/run_sweep.py \
  --config configs/experiments/llama31_mmlu_sweep.yaml
```

### Run complete sweep for both models

```bash
python scripts/run_sweep.py \
  --config configs/experiments/all_models_mmlu_sweep.yaml
```

### Debug run on small subset

```bash
python scripts/run_sweep.py \
  --config configs/experiments/all_models_mmlu_sweep.yaml \
  --max-samples 128 \
  --subjects abstract_algebra anatomy
```

### Run tests

```bash
pytest -q
```

### Run smoke test

```bash
python scripts/smoke_test.py
```

---

## Implementation notes and guardrails for Codex

### Model access

* Llama 3.1 may require accepted license terms and a Hugging Face token.
* Code should fail with a clear message if model access is unavailable.

### Memory constraints

* 8B models can be heavy.
* The code should support:

  * reduced evaluation subset for debugging
  * batch size override
  * auto device mapping
* For the main benchmark, avoid quantization unless explicitly requested.

### Determinism

* Set seeds for `random`, `numpy`, and `torch`.
* Avoid nondeterministic prompt formatting.
* Save resolved configs for each run.

### Fresh model per sparsity

* Reload model weights for each sparsity point.
* Do not prune incrementally across the sweep.
* Before each reload, explicitly delete the previous model and clear GPU memory:
  `del model; torch.cuda.empty_cache(); gc.collect()`
  Without this, loading a second 8B model will likely OOM.
* Optimization: instead of reloading from disk each time, deepcopy the original
  state dict before pruning and restore it per sparsity level to avoid repeated
  disk I/O (saves several minutes per model on slow storage).

### Logging

* Every run should write a dedicated `run.log`.
* Logs should include model id, sample counts, pruning stats, elapsed time, and any exceptions.

### Fail-safe behavior

* If one run fails, the sweep should continue to the next configuration unless `--fail-fast` is set.
* Failures should be summarized in the manifest.

---

## Makefile targets

Suggested `Makefile`:

```makefile
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

lint:
	ruff check .
	black --check .

test:
	pytest -q

smoke:
	python scripts/smoke_test.py

run-all:
	python scripts/run_sweep.py --config configs/experiments/all_models_mmlu_sweep.yaml

plot:
	python scripts/plot_results.py
```

---

## Manifest format

Create a top-level `manifest.json` per sweep.

It should contain:

* run name
* start time
* end time
* config hash
* completed runs
* failed runs
* skipped runs
* artifact paths

Example:

```json
{
  "run_name": "2026-04-16_all_models_mmlu",
  "completed": [
    {"model": "llama31_8b_instruct", "sparsity": 0},
    {"model": "llama31_8b_instruct", "sparsity": 10}
  ],
  "failed": [
    {"model": "qwen3_8b", "sparsity": 95, "reason": "CUDA out of memory"}
  ]
}
```

---

## Acceptance criteria

Codex implementation is done when all of the following are true:

1. `scripts/run_sweep.py` can run both requested models across all requested sparsities.
2. MMLU accuracy is computed and saved for every completed run.
3. Every run saves config, metrics, pruning stats, predictions, and logs.
4. Combined CSV and plot are generated.
5. Resume works correctly.
6. Unit tests pass.
7. Smoke integration tests pass.
8. README explains how to run everything.

---

## Nice-to-have extensions after the base version works

These are not part of the first implementation, but the code should be designed so they can be added later:

1. Per-layer sparsity analysis plots
2. Subject-wise MMLU breakdown plots
3. Structured and semi-structured pruning baselines
4. Calibration metrics in addition to accuracy
5. Pruning embeddings and lm_head as ablations
6. Support for `lm-evaluation-harness` backend as an alternative evaluator
7. WandB logging
8. Slurm launcher scripts for cluster sweeps

---

## Final instruction to Codex

Implement the repository exactly around this plan, with emphasis on:

* correctness
* reproducibility
* modularity
* clear artifacts
* robust tests

Do not take shortcuts that make the benchmark fragile.
Prefer explicit masks and explicit artifact saving over hidden framework magic.
Reload a clean model for every sparsity level.
Use MMLU choice scoring and accuracy as the core benchmark.

If a tradeoff is needed, favor reliability and testability over premature optimization.
