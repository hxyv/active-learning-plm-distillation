#!/usr/bin/env bash
set -euo pipefail

MINIFORGE_ROOT="${HOME}/miniforge3"
ENV_PREFIX="/opt/dlami/nvme/envs/esm3_gnn_distill"
PROJECT_ROOT="/opt/dlami/nvme/esm3_gnn_distill_baseline"

source "${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

conda env remove -p "${ENV_PREFIX}" -y >/dev/null 2>&1 || true
conda env create -p "${ENV_PREFIX}" -f "${PROJECT_ROOT}/envs/environment.yml"
conda activate "${ENV_PREFIX}"

# PyTorch + CUDA wheels and PyG. Pin torch so PyG extension wheels match (pt23 vs pt25 ABI).
pip install --upgrade pip
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch-geometric

echo "Environment ready at: ${ENV_PREFIX}"
