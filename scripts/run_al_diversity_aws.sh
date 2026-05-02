#!/usr/bin/env bash
# AWS diversity AL. Default: no graph embedding propagation.
#
# Direct:
#   ./scripts/run_al_diversity_aws.sh
#   PROPAGATE=1 ./scripts/run_al_diversity_aws.sh   # graph embedding propagation
#
# Long runs in tmux (recommended):
#   ./scripts/tmux_run_al_diversity_aws.sh
#   PROPAGATE=1 ./scripts/tmux_run_al_diversity_aws.sh
#   tmux attach -t al_diversity_aws
set -euo pipefail

PROJECT_ROOT="/opt/dlami/nvme/esm3_gnn_distill_baseline"
# Override: CONFIG=/path/to/custom.yaml ./scripts/run_al_diversity_aws.sh
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/al_aws_diversity_dispef_m.yaml}"
MINIFORGE_ROOT="${HOME}/miniforge3"
ENV_PREFIX="/opt/dlami/nvme/envs/esm3_gnn_distill"

STRATEGY="${STRATEGY:-diversity}"
RUN_ID="${AL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PROP_SUFFIX="noprop"
PROP_FLAG=()
if [[ "${PROPAGATE:-0}" == "1" ]]; then
  PROP_SUFFIX="prop"
  PROP_FLAG=(--Use_propagation)
fi
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/al/${STRATEGY}_${PROP_SUFFIX}_${RUN_ID}}"
RUN_NAME="${RUN_NAME:-aws_al_${STRATEGY}_${PROP_SUFFIX}_${RUN_ID}}"

RESUME_FLAG=""
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_FLAG="--resume"
  OUTPUT_DIR="${RESUME_OUTPUT_DIR:-${OUTPUT_DIR}}"
fi

mkdir -p "${PROJECT_ROOT}/outputs/al" "${PROJECT_ROOT}/logs"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# shellcheck source=/dev/null
source "${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

cd "${PROJECT_ROOT}"

exec python -m active_learning.al_loop \
  --config "${CONFIG}" \
  --strategy "${STRATEGY}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME}" \
  "${PROP_FLAG[@]}" \
  ${RESUME_FLAG}
