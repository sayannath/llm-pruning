#!/bin/bash
#SBATCH --job-name=structured-sft
#SBATCH --output=slurm/logs/structured_sft-%j.log
#SBATCH --error=slurm/logs/structured_sft-%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu-h100
#SBATCH --mail-user=sayan.nath@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

echo "========================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "Workdir:   $(pwd)"
echo "========================================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_env

set -euo pipefail

mkdir -p slurm/logs outputs/sft_runs

# ---------------------------------------------------------------------------
# Override variables (set at submission time via --export=ALL,VAR=val)
# ---------------------------------------------------------------------------
# CONFIG            Experiment config. Defaults to all-model structured SFT.
# FINETUNE_CONFIG   Fine-tuning config. Defaults to structured LoRA recovery.
# MAX_TRAIN_SAMPLES Cap training examples (e.g. 16 for smoke test).
# MAX_EVAL_SAMPLES  Cap evaluation examples.
# SPARSITIES        Space-separated sparsity points, e.g. "10 50".
# FAIL_FAST         Set to "1" to abort on first failed run.
# NO_RESUME         Set to "1" to re-run already-completed sparsity points.
#
# Examples:
#   sbatch slurm/run_all_models_structured_sft.slurm
#   sbatch --export=ALL,MAX_TRAIN_SAMPLES=16,MAX_EVAL_SAMPLES=16,FAIL_FAST=1 \
#     slurm/run_all_models_structured_sft.slurm
#   sbatch --export=ALL,SPARSITIES="10 50" \
#     slurm/run_all_models_structured_sft.slurm

CONFIG="${CONFIG:-configs/experiments/all_models_structured_sft.yaml}"
FINETUNE_CONFIG="${FINETUNE_CONFIG:-configs/finetuning/structured_lora_recovery.yaml}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
SPARSITIES="${SPARSITIES:-}"
FAIL_FAST="${FAIL_FAST:-0}"
NO_RESUME="${NO_RESUME:-0}"

CMD=(
  python scripts/run_sparse_finetune.py
  --config "$CONFIG"
  --finetune-config "$FINETUNE_CONFIG"
)

if [[ -n "$MAX_TRAIN_SAMPLES" ]]; then
  CMD+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
fi

if [[ -n "$MAX_EVAL_SAMPLES" ]]; then
  CMD+=(--max-eval-samples "$MAX_EVAL_SAMPLES")
fi

if [[ -n "$SPARSITIES" ]]; then
  read -r -a SPARSITY_ARRAY <<< "$SPARSITIES"
  CMD+=(--sparsities "${SPARSITY_ARRAY[@]}")
fi

if [[ "$FAIL_FAST" == "1" ]]; then
  CMD+=(--fail-fast)
fi

if [[ "$NO_RESUME" == "1" ]]; then
  CMD+=(--no-resume)
fi

echo "Experiment config:  $CONFIG"
echo "Finetune config:    $FINETUNE_CONFIG"
echo "Command:            ${CMD[*]}"
echo "========================================================"

"${CMD[@]}"

echo "========================================================"
echo "Finished:  $(date)"
echo "========================================================"
