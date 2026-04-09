#!/usr/bin/env bash
# Quick diversity AL smoke test (no --Use_propagation). Uses al_aws_diversity_dispef_m_smoke.yaml.
# After this succeeds, run ./scripts/run_al_diversity_aws.sh for the full-scale config.
set -euo pipefail

PROJECT_ROOT="/opt/dlami/nvme/esm3_gnn_distill_baseline"
CONFIG="${PROJECT_ROOT}/configs/al_aws_diversity_dispef_m_smoke.yaml"
MINIFORGE_ROOT="${HOME}/miniforge3"
ENV_PREFIX="/opt/dlami/nvme/envs/esm3_gnn_distill"

RUN_ID="${AL_RUN_ID:-smoke_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/al/diversity_smoke_${RUN_ID}}"
RUN_NAME="${RUN_NAME:-aws_al_diversity_smoke_${RUN_ID}}"

mkdir -p "${PROJECT_ROOT}/outputs/al" "${PROJECT_ROOT}/logs"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# shellcheck source=/dev/null
source "${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

cd "${PROJECT_ROOT}"

exec python -m active_learning.al_loop \
  --config "${CONFIG}" \
  --strategy diversity \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME}"
