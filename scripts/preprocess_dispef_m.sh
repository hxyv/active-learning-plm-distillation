#!/usr/bin/env bash
set -euo pipefail

MINIFORGE_ROOT="${HOME}/miniforge3"
ENV_PREFIX="/opt/dlami/nvme/envs/esm3_gnn_distill"
PROJECT_ROOT="/opt/dlami/nvme/esm3_gnn_distill_baseline"
RAW_ROOT="${PROJECT_ROOT}/data/raw/dispef"
PROCESSED_ROOT="${PROJECT_ROOT}/data/processed"
FETCH_UNIPROT="${FETCH_UNIPROT:-0}"

source "${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

EXTRA_ARGS=()
if [[ "${FETCH_UNIPROT}" == "1" ]]; then
  EXTRA_ARGS+=(--fetch-uniprot-sequences)
fi

python -m data.preprocess_dispef \
  --raw-root "${RAW_ROOT}" \
  --processed-root "${PROCESSED_ROOT}" \
  --dataset-name dispef_m \
  --val-fraction 0.1 \
  --seed 42 \
  "${EXTRA_ARGS[@]}"

echo "Preprocessing complete. Output: ${PROCESSED_ROOT}/dispef_m"
