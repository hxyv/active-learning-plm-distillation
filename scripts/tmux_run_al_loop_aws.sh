#!/usr/bin/env bash
# Start EMC (or other strategy) AL via run_al_loop_aws.sh in a new tmux session.
#
#   ./scripts/tmux_run_al_loop_aws.sh                    # detached; STRATEGY=emc
#   STRATEGY=random ./scripts/tmux_run_al_loop_aws.sh
#   RESUME=1 RESUME_OUTPUT_DIR=... ./scripts/tmux_run_al_loop_aws.sh
#   TMUX_ATTACH=1 ./scripts/tmux_run_al_loop_aws.sh
#
# Re-attach:   tmux attach -t "${TMUX_SESSION:-al_loop_aws}"
# List:        tmux ls
# Kill:        tmux kill-session -t "${TMUX_SESSION:-al_loop_aws}"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SESSION="${TMUX_SESSION:-al_loop_aws}"
STRATEGY="${STRATEGY:-emc}"
RESUME="${RESUME:-0}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install tmux or run: ./scripts/run_al_loop_aws.sh"
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
  exit 1
fi

# Optional env passthrough (set before invoking this script)
INNER="cd '${PROJECT_ROOT}' && export STRATEGY='${STRATEGY}' RESUME='${RESUME}'"
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

echo "Started tmux session: $SESSION  (STRATEGY=${STRATEGY} RESUME=${RESUME})"
echo "  attach: tmux attach -t $SESSION"
echo "  logs:   outputs/al/ and checkpoints/ for this run"

if [[ "${TMUX_ATTACH:-0}" == "1" ]]; then
  exec tmux attach -t "$SESSION"
fi
