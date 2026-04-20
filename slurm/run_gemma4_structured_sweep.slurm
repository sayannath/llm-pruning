#!/bin/bash
#SBATCH --job-name=mmlu-gemma4-structured
#SBATCH --output=slurm/logs/mmlu_gemma4_structured-%j.log
#SBATCH --error=slurm/logs/mmlu_gemma4_structured-%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu-h100
#SBATCH --mail-user=sayan.nath@ucalgary.ca
#SBATCH --mail-type=ALL

# Structured MLP-channel magnitude pruning sweep — Gemma-4-E4B-IT
#   Sparsity: 0, 10, 20, 30, 40, 50, 60, 70%  (8 points)
#   Est. time: ~20 min/run × 8 = ~2.7 h  → wall-time 12 h
#
#   Gemma-4 uses gate_proj/up_proj/down_proj — compatible with mlp_channel pruning.
#
# Submit:
#   sbatch slurm/run_gemma4_structured_sweep.slurm
#
# Override variables (--export=ALL,VAR=val):
#   MAX_SAMPLES  Cap on MMLU examples per sparsity point (e.g. 128 for a quick probe).
#   SUBJECTS     Space-separated MMLU subject names to restrict evaluation.
#   FAIL_FAST    Set to "1" to abort on the first failed sparsity run.
#   NO_RESUME    Set to "1" to re-run sparsity points already on disk.

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

# Gemma-4 is a gated model — set HF_TOKEN before submitting.
# export HF_TOKEN="..."

CONFIG="configs/experiments/gemma4_e4b_mmlu_structured_mlp_sweep.yaml"
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
echo "Method:    global_magnitude_structured (mlp_channel)"
echo "Config:    $CONFIG"
echo "Command:   ${CMD[*]}"
echo "========================================================"

"${CMD[@]}"

echo "========================================================"
echo "Finished:  $(date)"
echo "========================================================"
