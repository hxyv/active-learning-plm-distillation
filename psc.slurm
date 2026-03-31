#!/bin/bash
#SBATCH --job-name=esm3_gnn_distill
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -t 1-00:00:00
#SBATCH -A cis250233p
#SBATCH --output=/ocean/projects/cis250233p/%u/esm3_gnn_distill/logs/slurm/%j.out
#SBATCH --error=/ocean/projects/cis250233p/%u/esm3_gnn_distill/logs/slurm/%j.err
#SBATCH --export=ALL

WORKSPACE="/ocean/projects/cis250233p/${USER}/esm3_gnn_distill"
PROJECT_ROOT="${WORKSPACE}/active-learning-plm-distillation"
CONFIG="${PROJECT_ROOT}/configs/psc_dispef_m.yaml"

# create log dir if not present (SBATCH --output won't create it)
mkdir -p "${WORKSPACE}/logs/slurm"

module load anaconda3/2024.10-1
conda activate idl_diffusion_env

cd "${PROJECT_ROOT}"

python -m train.train \
  --config "${CONFIG}" \
  --run-name "psc_schake_distill_${SLURM_JOB_ID}"
