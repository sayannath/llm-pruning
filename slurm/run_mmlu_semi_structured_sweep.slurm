#!/bin/bash
#SBATCH --job-name=mmlu-semi-struct-sweep
#SBATCH --output=slurm/logs/mmlu_semi_structured_sweep-%j.log
#SBATCH --error=slurm/logs/mmlu_semi_structured_sweep-%j.log
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
# PATTERN     N:M pattern to run: "2_4" (default) or "4_8".
#             Ignored when CONFIG is set explicitly.
# CONFIG      Full path to a specific experiment config. Overrides PATTERN.
#             Defaults to configs/experiments/all_models_mmlu_semi_structured_<PATTERN>_sweep.yaml
# MAX_SAMPLES Cap on MMLU examples (e.g. 128 for a quick sensitivity probe).
# SUBJECTS    Space-separated MMLU subject names to restrict evaluation.
# FAIL_FAST   Set to "1" to abort on the first failed (model, sparsity) run.
# NO_RESUME   Set to "1" to re-run sparsity points already on disk.
#
# Examples — full two-model sweeps:
#   sbatch slurm/run_mmlu_semi_structured_sweep.slurm
#   sbatch --export=ALL,PATTERN=4_8 slurm/run_mmlu_semi_structured_sweep.slurm
#
# Examples — single-model sweeps:
#   sbatch --export=ALL,CONFIG=configs/experiments/llama31_mmlu_semi_structured_2_4_sweep.yaml \
#     slurm/run_mmlu_semi_structured_sweep.slurm
#   sbatch --export=ALL,CONFIG=configs/experiments/qwen3_mmlu_semi_structured_4_8_sweep.yaml \
#     slurm/run_mmlu_semi_structured_sweep.slurm
#
# Examples — 128-sample sensitivity probe (Stage C in the plan):
#   sbatch --export=ALL,PATTERN=2_4,MAX_SAMPLES=128 slurm/run_mmlu_semi_structured_sweep.slurm
#   sbatch --export=ALL,PATTERN=4_8,MAX_SAMPLES=128 slurm/run_mmlu_semi_structured_sweep.slurm
#
# Examples — single-subject smoke test:
#   sbatch --export=ALL,PATTERN=2_4,MAX_SAMPLES=32,SUBJECTS="abstract_algebra" \
#     slurm/run_mmlu_semi_structured_sweep.slurm

PATTERN="${PATTERN:-2_4}"
CONFIG="${CONFIG:-configs/experiments/all_models_mmlu_semi_structured_${PATTERN}_sweep.yaml}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SUBJECTS="${SUBJECTS:-}"
FAIL_FAST="${FAIL_FAST:-0}"
NO_RESUME="${NO_RESUME:-0}"

# Validate PATTERN when CONFIG was not set explicitly.
if [[ -z "${CONFIG_EXPLICIT+x}" ]]; then
  if [[ "$PATTERN" != "2_4" && "$PATTERN" != "4_8" ]]; then
    echo "ERROR: PATTERN must be '2_4' or '4_8', got: '$PATTERN'" >&2
    exit 1
  fi
fi

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
