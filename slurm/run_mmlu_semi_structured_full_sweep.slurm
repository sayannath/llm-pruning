#!/bin/bash
#SBATCH --job-name=mmlu-semi-struct-full
#SBATCH --output=slurm/logs/mmlu_semi_structured_full_%a-%j.log
#SBATCH --error=slurm/logs/mmlu_semi_structured_full_%a-%j.log
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu-h100
#SBATCH --mail-user=sayan.nath@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

# Job array layout:
#   task 0  →  2:4  (all_models_mmlu_semi_structured_2_4_sweep.yaml)
#   task 1  →  4:8  (all_models_mmlu_semi_structured_4_8_sweep.yaml)
#
# Submit:
#   sbatch slurm/run_mmlu_semi_structured_full_sweep.slurm
#
# Override variables (--export=ALL,VAR=val):
#   MAX_SAMPLES   Cap on MMLU examples per sparsity point (e.g. 128 for a probe).
#   SUBJECTS      Space-separated subject names to restrict evaluation.
#   FAIL_FAST     Set to "1" to abort on the first failed (model, sparsity) run.
#   NO_RESUME     Set to "1" to re-run sparsity points already on disk.

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

# Hugging Face gated models such as Llama 3.1 require an accepted license and token.
# export HF_TOKEN="..."

CONFIGS=(
  "configs/experiments/all_models_mmlu_semi_structured_2_4_sweep.yaml"
  "configs/experiments/all_models_mmlu_semi_structured_4_8_sweep.yaml"
)
PATTERNS=("2_4" "4_8")

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
PATTERN="${PATTERNS[$SLURM_ARRAY_TASK_ID]}"

MAX_SAMPLES="${MAX_SAMPLES:-}"
SUBJECTS="${SUBJECTS:-}"
FAIL_FAST="${FAIL_FAST:-0}"
NO_RESUME="${NO_RESUME:-0}"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config file not found: $CONFIG" >&2
  exit 1
fi

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

echo "Pattern:   $PATTERN"
echo "Config:    $CONFIG"
echo "Command:   ${CMD[*]}"
echo "========================================================"

"${CMD[@]}"

echo "========================================================"
echo "Finished:  $(date)"
echo "========================================================"
