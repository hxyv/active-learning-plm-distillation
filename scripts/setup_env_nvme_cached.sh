#!/usr/bin/env bash
# Wrapper: put conda/pip temp and caches on /opt/dlami/nvme so a small root disk (/) does not fill up.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_ROOT="/opt/dlami/nvme/.cache/esm3_gnn_distill_setup"
mkdir -p "${CACHE_ROOT}/tmp" "${CACHE_ROOT}/pip" "${CACHE_ROOT}/conda-pkgs"
export TMPDIR="${CACHE_ROOT}/tmp"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export CONDA_PKGS_DIRS="${CACHE_ROOT}/conda-pkgs"
exec bash "${SCRIPT_DIR}/setup_env_nvme.sh"
