#!/bin/bash
#SBATCH --job-name=gemma4-structured-sft
#SBATCH --output=slurm/logs/gemma4_structured_sft-%j.log
#SBATCH --error=slurm/logs/gemma4_structured_sft-%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu-l40
#SBATCH --mail-user=sayan.nath@ucalgary.ca
#SBATCH --mail-type=ALL

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
CONFIG="${CONFIG:-configs/experiments/gemma4_e4b_structured_sft.yaml}"
FINETUNE_CONFIG="${FINETUNE_CONFIG:-configs/finetuning/structured_lora_recovery.yaml}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
SPARSITIES="${SPARSITIES:-}"
FAIL_FAST="${FAIL_FAST:-0}"
# NO_RESUME="${NO_RESUME:-0}"  # resume logic removed

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

# if [[ "$NO_RESUME" == "1" ]]; then
#   CMD+=(--no-resume)
# fi

echo "Model:              Gemma-4-E4B-IT"
echo "Experiment config:  $CONFIG"
echo "Finetune config:    $FINETUNE_CONFIG"
echo "Command:            ${CMD[*]}"
echo "========================================================"

"${CMD[@]}"

echo "========================================================"
echo "Finished:  $(date)"
echo "========================================================"
