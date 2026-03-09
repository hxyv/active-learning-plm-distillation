#!/usr/bin/env bash
set -euo pipefail

MINIFORGE_ROOT="${HOME}/miniforge3"
ENV_PREFIX="/opt/dlami/nvme/envs/esm3_gnn_distill"
PROJECT_ROOT="/opt/dlami/nvme/esm3_gnn_distill_baseline"

source "${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

# Keep model/cache files off home directory.
export HF_HOME="/opt/dlami/nvme/esm3_gnn_distill_baseline/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

python -m teacher.generate_teacher_labels \
  --processed-root "${PROJECT_ROOT}/data/processed" \
  --dataset-name dispef_m \
  --teacher-cache-root "${PROJECT_ROOT}/cache/teacher" \
  --provider esm3 \
  --esm-backend auto \
  --split all \
  --device cuda
