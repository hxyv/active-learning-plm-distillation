#!/usr/bin/env bash
# Start full diversity AL in a new tmux session with graph embedding propagation (--Use_propagation).
#
#   ./scripts/tmux_run_al_diversity_aws_prop.sh              # detached; propagation on
#   TMUX_ATTACH=1 ./scripts/tmux_run_al_diversity_aws_prop.sh
#   PROPAGATE=0 ./scripts/tmux_run_al_diversity_aws_prop.sh  # override: no propagation
#
# Re-attach:   tmux attach -t "${TMUX_SESSION:-al_diversity_aws_prop}"
# List:        tmux ls
# Kill:        tmux kill-session -t "${TMUX_SESSION:-al_diversity_aws_prop}"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SESSION="${TMUX_SESSION:-al_diversity_aws_prop}"
PROPAGATE="${PROPAGATE:-1}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install tmux or run: PROPAGATE=1 bash scripts/run_al_diversity_aws.sh"
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
  exit 1
fi

# Window stays open after the run (exec bash) so you can scroll logs.
tmux new-session -d -s "$SESSION" bash -lc \
  "cd '${PROJECT_ROOT}' && export PROPAGATE='${PROPAGATE}' && bash scripts/run_al_diversity_aws.sh; ec=\$?; echo; echo \"=== finished (exit \$ec) ===\"; exec bash"

echo "Started tmux session: $SESSION  (PROPAGATE=${PROPAGATE})"
echo "  attach: tmux attach -t $SESSION"
echo "  logs:   also under outputs/al/ and checkpoints/ for this run"

if [[ "${TMUX_ATTACH:-0}" == "1" ]]; then
  exec tmux attach -t "$SESSION"
fi
