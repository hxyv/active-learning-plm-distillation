#!/bin/bash
#SBATCH --job-name=esm3_al_loop
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH -t 1-00:00:00
#SBATCH -A cis250233p
#SBATCH --output=/ocean/projects/cis250233p/xhu15/logs/slurm/%j.out
#SBATCH --error=/ocean/projects/cis250233p/xhu15/logs/slurm/%j.err
#SBATCH --export=ALL

PROJECT_ROOT="/ocean/projects/cis250233p/xhu15/active-learning-plm-distillation"
CONFIG="${PROJECT_ROOT}/configs/al_psc_dispef_m.yaml"

# Acquisition strategy: random (passive baseline), mc_dropout, emc, diversity
STRATEGY="${STRATEGY:-random}"

# Output directory for this AL run (one directory per strategy/seed)
OUTPUT_DIR="/ocean/projects/cis250233p/xhu15/outputs/al/${STRATEGY}_${SLURM_JOB_ID}"

# Resume an existing run by setting RESUME=1 when submitting:
#   sbatch --export=ALL,STRATEGY=random,RESUME=1 slurm/al_loop.slurm
RESUME_FLAG=""
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_FLAG="--resume"
  # When resuming, point to the existing output directory
  OUTPUT_DIR="${RESUME_OUTPUT_DIR:-${OUTPUT_DIR}}"
fi

mkdir -p "/ocean/projects/cis250233p/xhu15/logs/slurm"

module load anaconda3/2024.10-1
conda activate /ocean/projects/cis250233p/xhu15/envs/esm3_gnn_distill

cd "${PROJECT_ROOT}"

python -m active_learning.al_loop \
  --config   "${CONFIG}" \
  --strategy "${STRATEGY}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "psc_al_${STRATEGY}_${SLURM_JOB_ID}" \
  ${RESUME_FLAG}
