#!/bin/bash
#SBATCH --job-name=mmlu-prune-sweep
#SBATCH --output=slurm/logs/mmlu_prune_sweep-%j.log
#SBATCH --error=slurm/logs/mmlu_prune_sweep-%j.log
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

# Defaults run the complete two-model sweep from the plan. Override at submission time:
#   sbatch --export=ALL,CONFIG=configs/experiments/llama31_mmlu_sweep.yaml slurm/run_mmlu_pruning_sweep.slurm
#   sbatch --export=ALL,MAX_SAMPLES=128,SUBJECTS="abstract_algebra anatomy" slurm/run_mmlu_pruning_sweep.slurm
CONFIG="${CONFIG:-configs/experiments/all_models_mmlu_sweep.yaml}"
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

echo "Command: ${CMD[*]}"
echo "========================================================"

"${CMD[@]}"

echo "========================================================"
echo "Finished:  $(date)"
echo "========================================================"
