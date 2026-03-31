#!/bin/bash
#SBATCH --job-name=dispef_preprocess
#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH -t 4:00:00
#SBATCH -A cis250233p
#SBATCH --output=/ocean/projects/cis250233p/xhu15/logs/slurm/%j.out
#SBATCH --error=/ocean/projects/cis250233p/xhu15/logs/slurm/%j.err
#SBATCH --export=ALL

PROJECT_ROOT="/ocean/projects/cis250233p/xhu15/active-learning-plm-distillation"

mkdir -p "/ocean/projects/cis250233p/xhu15/logs/slurm"

module load anaconda3/2024.10-1
conda activate /ocean/projects/cis250233p/xhu15/envs/esm3_gnn_distill

cd "${PROJECT_ROOT}"

python -m data.preprocess_dispef \
  --raw-root       /ocean/projects/cis250233p/xhu15/data/raw/dispef \
  --processed-root /ocean/projects/cis250233p/xhu15/data/processed \
  --dataset-name   dispef_m \
  --val-fraction   0.1 \
  --seed           42
