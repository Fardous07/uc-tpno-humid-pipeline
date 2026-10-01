#!/usr/bin/env bash
# Start / stop the Stage 3 job queue (see scripts/stage3_queue.py).
#   bash scripts/stage3_queue.sh start [N_SLOTS]   # default: cores - 1
#   bash scripts/stage3_queue.sh status
#   bash scripts/stage3_queue.sh kill
set -euo pipefail

PROJECT_ROOT="/home/fardo/uc_tpno_humid_pipeline/uc_tpno_humid_pipeline"
PY="${PROJECT_ROOT}/thesis_env/bin/python"
QUEUE_PY="${PROJECT_ROOT}/scripts/stage3_queue.py"
LOG="${PROJECT_ROOT}/data/processed/adsorption_v3/logs/stage3_queue.log"
SESSION="gcmc_queue"

case "${1:-status}" in
    start)
        N_SLOTS="${2:-$(( $(nproc --all) - 1 ))}"
        if tmux has-session -t "${SESSION}" 2>/dev/null; then
            echo "Queue already running (tmux session ${SESSION})."
            exit 1
        fi
        tmux new-session -d -s "${SESSION}" \
            "setsid ${PY} ${QUEUE_PY} run ${N_SLOTS} >> ${LOG} 2>&1"
        sleep 3
        echo "Queue started with ${N_SLOTS} slots (tmux session ${SESSION})."
        echo "  Status : bash scripts/stage3_queue.sh status"
        echo "  Stop   : bash scripts/stage3_queue.sh kill"
        ;;
    status)
        "${PY}" "${QUEUE_PY}" status
        ;;
    kill)
        "${PY}" "${QUEUE_PY}" kill
        tmux kill-session -t "${SESSION}" 2>/dev/null || true
        echo "Done. Resume later with: bash scripts/stage3_queue.sh start"
        ;;
    *)
        echo "Usage: $0 {start [N_SLOTS]|status|kill}"
        exit 1
        ;;
esac
