#!/usr/bin/env bash
# AWS / local GPU runner for active learning (EMC by default).
# PSC Bridges uses slurm/al_loop.slurm + configs/al_psc_dispef_m.yaml — do not remove or merge.
set -euo pipefail

PROJECT_ROOT="/opt/dlami/nvme/esm3_gnn_distill_baseline"
# Override: CONFIG=/path/to/custom.yaml ./scripts/run_al_loop_aws.sh
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/al_aws_dispef_m.yaml}"
MINIFORGE_ROOT="${HOME}/miniforge3"
ENV_PREFIX="/opt/dlami/nvme/envs/esm3_gnn_distill"

# Override with: STRATEGY=random ./scripts/run_al_loop_aws.sh
# Choices: random, mc_dropout, emc, diversity
STRATEGY="${STRATEGY:-emc}"

# Unique id for this invocation (override: AL_RUN_ID=myexp1)
RUN_ID="${AL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/al/${STRATEGY}_${RUN_ID}}"
RUN_NAME="${RUN_NAME:-aws_al_${STRATEGY}_${RUN_ID}}"

RESUME_FLAG=""
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_FLAG="--resume"
  OUTPUT_DIR="${RESUME_OUTPUT_DIR:-${OUTPUT_DIR}}"
fi

mkdir -p "${PROJECT_ROOT}/outputs/al" "${PROJECT_ROOT}/logs"

# Reduce allocator fragmentation on long runs (optional but helps after OOM retries)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Conda env from scripts/setup_env_nvme.sh (or setup_env_nvme_cached.sh)
# shellcheck source=/dev/null
source "${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

cd "${PROJECT_ROOT}"

exec python -m active_learning.al_loop \
  --config "${CONFIG}" \
  --strategy "${STRATEGY}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME}" \
  ${RESUME_FLAG}
