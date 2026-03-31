#!/bin/bash
#SBATCH --job-name=esm3_teacher
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -t 1-00:00:00
#SBATCH -A cis250233p
#SBATCH --output=/ocean/projects/cis250233p/xhu15/logs/slurm/%j.out
#SBATCH --error=/ocean/projects/cis250233p/xhu15/logs/slurm/%j.err
#SBATCH --export=ALL

PROJECT_ROOT="/ocean/projects/cis250233p/xhu15/active-learning-plm-distillation"

mkdir -p "/ocean/projects/cis250233p/xhu15/logs/slurm"

module load anaconda3/2024.10-1
module load cuda/12.4.0
source activate esm3_gnn_distill

cd "${PROJECT_ROOT}"

# Set to "dispef_m" or "dispef_s"
DATASET="${DATASET:-dispef_m}"

python -m teacher.generate_teacher_labels \
  --processed-root     /ocean/projects/cis250233p/xhu15/data/processed \
  --dataset-name       "${DATASET}" \
  --teacher-cache-root /ocean/projects/cis250233p/xhu15/cache/teacher \
  --provider esm3 --esm-backend local --split all --device cuda
