#!/bin/bash
#SBATCH --job-name=esm3_gnn_eval
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -t 2:00:00
#SBATCH -A cis250233p
#SBATCH --output=/ocean/projects/cis250233p/xhu15/logs/slurm/%j.out
#SBATCH --error=/ocean/projects/cis250233p/xhu15/logs/slurm/%j.err
#SBATCH --export=ALL

PROJECT_ROOT="/ocean/projects/cis250233p/xhu15/active-learning-plm-distillation"

mkdir -p "/ocean/projects/cis250233p/xhu15/logs/slurm"

module load anaconda3/2024.10-1
conda activate /ocean/projects/cis250233p/xhu15/envs/esm3_gnn_distill

cd "${PROJECT_ROOT}"

# Pass RUN_NAME when submitting: sbatch --export=ALL,RUN_NAME=<run> slurm/eval.slurm
if [[ -z "${RUN_NAME}" ]]; then
  echo "ERROR: RUN_NAME is not set. Submit with: sbatch --export=ALL,RUN_NAME=<run_dir> slurm/eval.slurm"
  exit 1
fi

python -m eval.evaluate \
  --config     "${PROJECT_ROOT}/configs/psc_dispef_m.yaml" \
  --checkpoint /ocean/projects/cis250233p/xhu15/checkpoints/${RUN_NAME}/best.pt \
  --split      test \
  --output-dir /ocean/projects/cis250233p/xhu15/outputs/eval/${RUN_NAME}
