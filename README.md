# LLM Pruning MMLU

Reproducible global unstructured pruning sweeps for causal LMs evaluated with
zero-shot MMLU multiple-choice accuracy.

## Setup

```bash
pip install -r requirements-dev.txt
```

Llama 3.1 access may require accepting the Hugging Face license and setting
`HF_TOKEN`.

## Validate

```bash
python scripts/validate_env.py
pytest -q
```

## Run One Sparsity Point

```bash
python scripts/evaluate_model.py \
  --config configs/experiments/llama31_mmlu_sweep.yaml \
  --sparsity 50 \
  --max-samples 128
```

## Run Sweep

```bash
python scripts/run_sweep.py \
  --config configs/experiments/all_models_mmlu_sweep.yaml
```

Outputs are written under `outputs/runs/<run_id>/`, including resolved configs,
metrics, predictions, pruning stats, combined CSV/JSONL, and plots.
