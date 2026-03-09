#!/usr/bin/env bash
set -euo pipefail

MINIFORGE_ROOT="${HOME}/miniforge3"
ENV_PREFIX="/opt/dlami/nvme/envs/esm3_gnn_distill"
PROJECT_ROOT="/opt/dlami/nvme/esm3_gnn_distill_baseline"

source "${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

python -m train.train \
  --config "${PROJECT_ROOT}/configs/paper_dispef_m.yaml" \
  --run-name "paper_schake_distill"
