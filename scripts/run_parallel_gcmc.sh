#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/fardo/uc_tpno_humid_pipeline/uc_tpno_humid_pipeline"
VENV="${PROJECT_ROOT}/thesis_env/bin/activate"
CIF_DIR="${PROJECT_ROOT}/data/intermediate/cifs_sanitized"
OUT_DIR="${PROJECT_ROOT}/data/processed/adsorption"
CHUNK_DIR="${PROJECT_ROOT}/data/intermediate/cif_chunks"
LOG_DIR="${OUT_DIR}/logs"
RASPA_BIN="/home/fardo/miniconda3/envs/raspa-env/bin/simulate"
RASPA_DATA="/home/fardo/miniconda3/envs/raspa-env"
SCRIPT_PY="${PROJECT_ROOT}/scripts/04_run_simulations.py"
PID_DIR="/tmp/gcmc_pids"
N_CYCLES=1000
TEMPERATURES="298.15,313.15,333.15"
PRESSURES="0.1,1.0,5.0,10.0"
RHS="0.0,0.05,0.15"
DRY_CO2_FRAC=0.15

ACTION="${1:-launch}"

if [[ "$ACTION" == "--status" ]]; then
    echo "========================================"
    echo "GCMC WORKER STATUS"
    echo "========================================"
    running=0
    finished=0
    for pid_file in "${PID_DIR}"/worker_*.pid; do
        [[ -f "$pid_file" ]] || continue
        worker=$(basename "$pid_file" .pid)
        pid=$(cat "$pid_file")
        log="${LOG_DIR}/${worker}.log"
        done_count=0
        if [[ -f "$log" ]]; then
            done_count=$(grep -c "Simulation complete\|saved to" "$log" 2>/dev/null || true)
        fi
        if kill -0 "$pid" 2>/dev/null; then
            echo "  ${worker}: PID=${pid} | RUNNING | done~${done_count}"
            running=$((running+1))
        else
            echo "  ${worker}: PID=${pid} | FINISHED | done~${done_count}"
            finished=$((finished+1))
        fi
    done
    n=$(find "${OUT_DIR}" -name "*.parquet" 2>/dev/null | wc -l)
    echo "----------------------------------------"
    echo "  Running: ${running} | Finished: ${finished}"
    echo "  Parquet files done: ${n}/998"
    echo "========================================"
    exit 0
fi

if [[ "$ACTION" == "--kill" ]]; then
    echo "Killing all workers..."
    for pid_file in "${PID_DIR}"/worker_*.pid; do
        [[ -f "$pid_file" ]] || continue
        pid=$(cat "$pid_file")
        worker=$(basename "$pid_file" .pid)
        if kill "$pid" 2>/dev/null; then
            echo "  Killed ${worker}"
        else
            echo "  ${worker} already stopped"
        fi
    done
    tmux kill-server 2>/dev/null || true
    echo "Done."
    exit 0
fi

# --- LAUNCH ---
if [[ -n "${1:-}" ]] && [[ "$1" =~ ^[0-9]+$ ]]; then
    N_WORKERS=$1
else
    N_CORES=$(nproc --all)
    N_WORKERS=$(( N_CORES > 1 ? N_CORES - 1 : 1 ))
    N_WORKERS=$(( N_WORKERS > 16 ? 16 : N_WORKERS ))
fi

cd "${PROJECT_ROOT}"
echo "========================================"
echo "PARALLEL GCMC LAUNCHER - ${N_WORKERS} workers"
echo "========================================"

mapfile -t ALL_CIFS < <(ls "${CIF_DIR}"/*.cif 2>/dev/null | sort)
N_TOTAL=${#ALL_CIFS[@]}
if [[ ${N_TOTAL} -eq 0 ]]; then
    echo "ERROR: No CIFs found in ${CIF_DIR}"
    exit 1
fi
if [[ ${N_WORKERS} -gt ${N_TOTAL} ]]; then
    N_WORKERS=${N_TOTAL}
fi
CHUNK_SIZE=$(( (N_TOTAL + N_WORKERS - 1) / N_WORKERS ))
echo "  CIFs: ${N_TOTAL} | Chunk: ${CHUNK_SIZE} each"

mkdir -p "${LOG_DIR}" "${PID_DIR}"
rm -rf "${CHUNK_DIR}"
mkdir -p "${CHUNK_DIR}"

for (( w=0; w<N_WORKERS; w++ )); do
    chunk_path="${CHUNK_DIR}/worker_${w}"
    mkdir -p "${chunk_path}"
    start=$(( w * CHUNK_SIZE ))
    end=$(( start + CHUNK_SIZE ))
    if [[ ${end} -gt ${N_TOTAL} ]]; then
        end=${N_TOTAL}
    fi
    for (( i=start; i<end; i++ )); do
        ln -sf "${ALL_CIFS[$i]}" "${chunk_path}/$(basename "${ALL_CIFS[$i]}")"
    done
    count=$(( end - start ))
    echo "  Worker ${w}: ${count} CIFs"
done

tmux kill-server 2>/dev/null || true
sleep 1

for (( w=0; w<N_WORKERS; w++ )); do
    SESSION="gcmc_worker_${w}"
    CHUNK_PATH="${CHUNK_DIR}/worker_${w}"
    WORKER_OUT="${OUT_DIR}/worker_${w}"
    LOG_FILE="${LOG_DIR}/worker_${w}.log"
    PID_FILE="${PID_DIR}/worker_${w}.pid"
    mkdir -p "${WORKER_OUT}"

    tmux new-session -d -s "${SESSION}"
    tmux send-keys -t "${SESSION}" "source ${VENV} && nohup python ${SCRIPT_PY} --cif-dir ${CHUNK_PATH} --output-dir ${WORKER_OUT} --n-cycles ${N_CYCLES} --temperatures '${TEMPERATURES}' --pressures '${PRESSURES}' --rhs '${RHS}' --dry-co2-frac ${DRY_CO2_FRAC} --raspa-path ${RASPA_BIN} --raspa-data-dir ${RASPA_DATA} > ${LOG_FILE} 2>&1 & echo \$! > ${PID_FILE} && echo Worker ${w} started PID: \$(cat ${PID_FILE})" ENTER
    echo "  Launched ${SESSION}"
done

sleep 3
echo ""
echo "All ${N_WORKERS} workers launched."
echo "  Status : bash scripts/run_parallel_gcmc.sh --status"
echo "  Kill   : bash scripts/run_parallel_gcmc.sh --kill"
echo "  Log    : tail -f ${LOG_DIR}/worker_0.log"
echo "  Count  : find data/processed/adsorption/worker_* -name '*.parquet' | wc -l"
echo "  Merge  : python scripts/merge_simulation_results.py"
