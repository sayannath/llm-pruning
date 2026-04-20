#!/bin/bash
#SBATCH --job-name=mmlu-gemma4-semi-struct
#SBATCH --output=slurm/logs/mmlu_gemma4_semi_structured_%a-%j.log
#SBATCH --error=slurm/logs/mmlu_gemma4_semi_structured_%a-%j.log
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu-h100
#SBATCH --mail-user=sayan.nath@ucalgary.ca
#SBATCH --mail-type=ALL

# Semi-structured N:M magnitude pruning sweep — Gemma-4-E4B-IT
#   Sparsity: 0, 50%  (2 points × 2 patterns = 4 runs total)
#   Est. time: ~20 min/run × 2 = ~40 min/task  → wall-time 6 h
#
# Job array layout (both tasks run in parallel):
#   task 0  →  2:4  pattern
#   task 1  →  4:8  pattern
#
# Submit:
#   sbatch slurm/run_gemma4_semi_structured_sweep.slurm
#
# Single-pattern only:
#   sbatch --array=0 slurm/run_gemma4_semi_structured_sweep.slurm   # 2:4 only
#   sbatch --array=1 slurm/run_gemma4_semi_structured_sweep.slurm   # 4:8 only
#
# Override variables (--export=ALL,VAR=val):
#   MAX_SAMPLES  Cap on MMLU examples per sparsity point.
#   SUBJECTS     Space-separated MMLU subject names to restrict evaluation.
#   FAIL_FAST    Set to "1" to abort on the first failed sparsity run.
#   NO_RESUME    Set to "1" to re-run sparsity points already on disk.

echo "========================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Array ID:  $SLURM_ARRAY_TASK_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "Workdir:   $(pwd)"
echo "========================================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_env

set -euo pipefail

mkdir -p slurm/logs outputs/runs outputs/summaries outputs/plots

# Gemma-4 is a gated model — set HF_TOKEN before submitting.
# export HF_TOKEN="..."

CONFIGS=(
  "configs/experiments/gemma4_e4b_mmlu_semi_structured_2_4_sweep.yaml"
  "configs/experiments/gemma4_e4b_mmlu_semi_structured_4_8_sweep.yaml"
)
PATTERNS=("2:4" "4:8")

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
PATTERN="${PATTERNS[$SLURM_ARRAY_TASK_ID]}"

MAX_SAMPLES="${MAX_SAMPLES:-}"
SUBJECTS="${SUBJECTS:-}"
FAIL_FAST="${FAIL_FAST:-0}"
NO_RESUME="${NO_RESUME:-0}"

CMD=(python scripts/run_sweep.py --config "$CONFIG")

[[ -n "$MAX_SAMPLES" ]] && CMD+=(--max-samples "$MAX_SAMPLES")

if [[ -n "$SUBJECTS" ]]; then
  read -r -a SUBJECT_ARRAY <<< "$SUBJECTS"
  CMD+=(--subjects "${SUBJECT_ARRAY[@]}")
fi

[[ "$FAIL_FAST" == "1" ]] && CMD+=(--fail-fast)
[[ "$NO_RESUME" == "1" ]] && CMD+=(--no-resume)

echo "Model:     google/gemma-4-E4B-it"
echo "Method:    global_magnitude_semi_structured  pattern=$PATTERN"
echo "Config:    $CONFIG"
echo "Command:   ${CMD[*]}"
echo "========================================================"

"${CMD[@]}"

echo "========================================================"
echo "Finished:  $(date)"
echo "========================================================"
