#!/bin/bash
#SBATCH --job-name=esm3_mc_smoke
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:15:00
#SBATCH -A cis250233p
#SBATCH --output=/ocean/projects/cis250233p/xhu15/logs/slurm/%j.out
#SBATCH --error=/ocean/projects/cis250233p/xhu15/logs/slurm/%j.err
#SBATCH --export=ALL

# Smoke test for the inter-conv MC Dropout patch in SchakeDistillModel.
# Runs scripts/smoke_test_mc_dropout.py on a single GPU batch.  ~1-2 min wall.
#
# Submit:  sbatch slurm/smoke_test.slurm
# Resume safety: this job writes no state; safe to re-run any time.

PROJECT_ROOT="/ocean/projects/cis250233p/xhu15/active-learning-plm-distillation"
CONFIG="${PROJECT_ROOT}/configs/al_psc_dispef_m.yaml"

mkdir -p "/ocean/projects/cis250233p/xhu15/logs/slurm"

module load anaconda3/2024.10-1
conda activate /ocean/projects/cis250233p/xhu15/envs/esm3_gnn_distill

cd "${PROJECT_ROOT}"

set -euo pipefail

python -u scripts/smoke_test_mc_dropout.py \
  --config    "${CONFIG}" \
  --n-passes  20 \
  --batch-size 4
