#!/usr/bin/env bash
set -euo pipefail

MINIFORGE_ROOT="${HOME}/miniforge3"
ENV_PREFIX="/opt/dlami/nvme/envs/esm3_gnn_distill"
PROJECT_ROOT="/opt/dlami/nvme/esm3_gnn_distill_baseline"

source "${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

# Replace with the checkpoint you want to evaluate.
CHECKPOINT="${PROJECT_ROOT}/checkpoints/<RUN_DIR>/best.pt"
EVAL_OUT="${PROJECT_ROOT}/outputs/eval/<RUN_DIR>"

python -m eval.evaluate \
  --config "${PROJECT_ROOT}/configs/baseline_dispef_m.yaml" \
  --checkpoint "${CHECKPOINT}" \
  --split test \
  --batch-size 16 \
  --save-predictions \
  --output-dir "${EVAL_OUT}"
