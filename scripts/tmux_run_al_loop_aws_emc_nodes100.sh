#!/usr/bin/env bash
# EMC AL with al_aws_dispef_m_emc_nodes100.yaml (emc_max_nodes_per_graph: 100) in a new tmux session.
#
#   ./scripts/tmux_run_al_loop_aws_emc_nodes100.sh
#   STRATEGY=random CONFIG=/path/to/other.yaml ./scripts/tmux_run_al_loop_aws_emc_nodes100.sh
#   RESUME=1 RESUME_OUTPUT_DIR=... ./scripts/tmux_run_al_loop_aws_emc_nodes100.sh
#   TMUX_ATTACH=1 ./scripts/tmux_run_al_loop_aws_emc_nodes100.sh
#
# Re-attach:   tmux attach -t "${TMUX_SESSION:-al_loop_aws_emc_nodes100}"
# List:        tmux ls
# Kill:        tmux kill-session -t "${TMUX_SESSION:-al_loop_aws_emc_nodes100}"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SESSION="${TMUX_SESSION:-al_loop_aws_emc_nodes100}"
STRATEGY="${STRATEGY:-emc}"
RESUME="${RESUME:-0}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/al_aws_dispef_m_emc_nodes100.yaml}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install tmux or run: CONFIG='${CONFIG}' ./scripts/run_al_loop_aws.sh"
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
  exit 1
fi

INNER="cd '${PROJECT_ROOT}' && export CONFIG='${CONFIG}' STRATEGY='${STRATEGY}' RESUME='${RESUME}'"
if [[ -n "${AL_RUN_ID:-}" ]]; then
  INNER+=" && export AL_RUN_ID='${AL_RUN_ID}'"
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  INNER+=" && export OUTPUT_DIR='${OUTPUT_DIR}'"
fi
if [[ -n "${RUN_NAME:-}" ]]; then
  INNER+=" && export RUN_NAME='${RUN_NAME}'"
fi
if [[ -n "${RESUME_OUTPUT_DIR:-}" ]]; then
  INNER+=" && export RESUME_OUTPUT_DIR='${RESUME_OUTPUT_DIR}'"
fi
INNER+=" && bash scripts/run_al_loop_aws.sh; ec=\$?; echo; echo \"=== finished (exit \$ec) ===\"; exec bash"

tmux new-session -d -s "$SESSION" bash -lc "$INNER"

echo "Started tmux session: $SESSION  (CONFIG=${CONFIG} STRATEGY=${STRATEGY} RESUME=${RESUME})"
echo "  attach: tmux attach -t $SESSION"
echo "  logs:   outputs/al/ and checkpoints/ for this run"

if [[ "${TMUX_ATTACH:-0}" == "1" ]]; then
  exec tmux attach -t "$SESSION"
fi
