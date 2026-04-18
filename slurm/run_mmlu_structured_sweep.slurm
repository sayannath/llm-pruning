#!/bin/bash
#SBATCH --job-name=mmlu-structured-sweep
#SBATCH --output=slurm/logs/mmlu_structured_sweep-%j.log
#SBATCH --error=slurm/logs/mmlu_structured_sweep-%j.log
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

mkdir -p slurm/logs outputs/runs outputs/summaries outputs/plots

# Hugging Face gated models such as Llama 3.1 require an accepted license and token.
# Export HF_TOKEN before submitting, or uncomment and set it here in a secure environment.
# export HF_TOKEN="..."

# ---------------------------------------------------------------------------
# Override variables (set at submission time via --export=ALL,VAR=val)
# ---------------------------------------------------------------------------
# CONFIG      Config file to run. Defaults to the full two-model MLP sweep.
# MAX_SAMPLES Cap on MMLU examples (e.g. 128 for a quick sanity check).
# SUBJECTS    Space-separated MMLU subject names to restrict evaluation.
# FAIL_FAST   Set to "1" to abort on the first failed (model, sparsity) run.
# NO_RESUME   Set to "1" to re-run sparsity points already on disk.
#
# Examples:
#   sbatch slurm/run_mmlu_structured_sweep.slurm
#   sbatch --export=ALL,CONFIG=configs/experiments/llama31_mmlu_structured_mlp_sweep.yaml \
#     slurm/run_mmlu_structured_sweep.slurm
#   sbatch --export=ALL,MAX_SAMPLES=128,FAIL_FAST=1 \
#     slurm/run_mmlu_structured_sweep.slurm

CONFIG="${CONFIG:-configs/experiments/all_models_mmlu_structured_mlp_sweep.yaml}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SUBJECTS="${SUBJECTS:-}"
FAIL_FAST="${FAIL_FAST:-0}"
NO_RESUME="${NO_RESUME:-0}"

CMD=(python scripts/run_sweep.py --config "$CONFIG")

if [[ -n "$MAX_SAMPLES" ]]; then
  CMD+=(--max-samples "$MAX_SAMPLES")
fi

if [[ -n "$SUBJECTS" ]]; then
  read -r -a SUBJECT_ARRAY <<< "$SUBJECTS"
  CMD+=(--subjects "${SUBJECT_ARRAY[@]}")
fi

if [[ "$FAIL_FAST" == "1" ]]; then
  CMD+=(--fail-fast)
fi

if [[ "$NO_RESUME" == "1" ]]; then
  CMD+=(--no-resume)
fi

echo "Config:    $CONFIG"
echo "Command:   ${CMD[*]}"
echo "========================================================"

"${CMD[@]}"

echo "========================================================"
echo "Finished:  $(date)"
echo "========================================================"
