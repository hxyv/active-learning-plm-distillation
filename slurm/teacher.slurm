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

# Store HF model weights on ocean (not ~/.cache which has limited quota on PSC)
export HF_HOME="/ocean/projects/cis250233p/xhu15/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

module load anaconda3/2024.10-1
module load cuda/12.4.0
conda activate /ocean/projects/cis250233p/xhu15/envs/esm3_gnn_distill

cd "${PROJECT_ROOT}"

# Set DATASET and sharding via --export when submitting:
#   sbatch --export=ALL,DATASET=dispef_m,NUM_SHARDS=2,SHARD_ID=0 slurm/teacher.slurm
DATASET="${DATASET:-dispef_m}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-0}"

python -m teacher.generate_teacher_labels \
  --processed-root     /ocean/projects/cis250233p/xhu15/data/processed \
  --dataset-name       "${DATASET}" \
  --teacher-cache-root /ocean/projects/cis250233p/xhu15/cache/teacher \
  --provider esm3 --esm-backend local --split all --device cuda \
  --num-shards "${NUM_SHARDS}" \
  --shard-id   "${SHARD_ID}"
