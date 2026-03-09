#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/opt/dlami/nvme/esm3_gnn_distill_baseline"
RAW_ROOT="${PROJECT_ROOT}/data/raw/dispef"
ARCHIVE_PATH="${RAW_ROOT}/zenodo_13755810_files_archive.zip"

mkdir -p "${RAW_ROOT}"

curl -L "https://zenodo.org/api/records/13755810/files-archive" -o "${ARCHIVE_PATH}"
unzip -o "${ARCHIVE_PATH}" -d "${RAW_ROOT}"

echo "DISPEF archive downloaded and unpacked to: ${RAW_ROOT}"
