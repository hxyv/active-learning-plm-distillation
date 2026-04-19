#!/bin/bash
#SBATCH --job-name=esm3_bench_latency
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --cpus-per-task=4
#SBATCH -t 01:00:00
#SBATCH -A cis250233p
#SBATCH --output=/ocean/projects/cis250233p/xhu15/logs/slurm/%j.out
#SBATCH --error=/ocean/projects/cis250233p/xhu15/logs/slurm/%j.err
#SBATCH --export=ALL

# Per-query latency benchmark for ESM3 teacher variants.  Motivates active
# learning: AL saves (pool_size - al_final) teacher queries, so a larger
# per-query cost ⇒ larger absolute saving.
#
# Submit:
#   # local 1.4B only
#   sbatch slurm/bench_teacher.slurm
#   # multiple variants (requires ESM_API_TOKEN for Forge models)
#   sbatch --export=ALL,MODELS="esm3_sm_open_v1 esm3-medium-2024-03",BACKENDS="local forge" \
#          slurm/bench_teacher.slurm

PROJECT_ROOT="/ocean/projects/cis250233p/xhu15/active-learning-plm-distillation"
CONFIG="${PROJECT_ROOT}/configs/al_psc_dispef_m.yaml"

MODELS="${MODELS:-esm3_sm_open_v1}"
BACKENDS="${BACKENDS:-auto}"
N_PROTEINS="${N_PROTEINS:-30}"
USE_STRUCTURE_FLAG=""
if [[ "${USE_STRUCTURE:-0}" == "1" ]]; then
  USE_STRUCTURE_FLAG="--use-structure"
fi

# HF cache needs to live on /ocean (jet/home is quota-limited).  Same mapping
# as slurm/teacher.slurm.
export HF_HOME="/ocean/projects/cis250233p/xhu15/cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "/ocean/projects/cis250233p/xhu15/logs/slurm"

module load anaconda3/2024.10-1
conda activate /ocean/projects/cis250233p/xhu15/envs/esm3_gnn_distill

cd "${PROJECT_ROOT}"

OUT="/ocean/projects/cis250233p/xhu15/outputs/bench/teacher_latency_${SLURM_JOB_ID}.json"
mkdir -p "$(dirname "${OUT}")"

set -euo pipefail

python -u scripts/bench_teacher_latency.py \
  --config "${CONFIG}" \
  --models ${MODELS} \
  --backends ${BACKENDS} \
  --n-proteins "${N_PROTEINS}" \
  --warmup 3 \
  ${USE_STRUCTURE_FLAG} \
  --output "${OUT}"
