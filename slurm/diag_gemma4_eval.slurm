#!/bin/bash
#SBATCH --job-name=diag-gemma4-eval
#SBATCH --output=slurm/logs/diag_gemma4_eval-%j.log
#SBATCH --error=slurm/logs/diag_gemma4_eval-%j.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0-00:30:00
#SBATCH --partition=gpu-l40
#SBATCH --mail-user=sayan.nath@ucalgary.ca
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_env
set -euo pipefail

echo "========================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "========================================================"

python scripts/diag_gemma4_eval.py

echo "========================================================"
echo "Finished:  $(date)"
echo "========================================================"
