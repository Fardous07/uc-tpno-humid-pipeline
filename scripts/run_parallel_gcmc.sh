#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════
# run_parallel_gcmc.sh — Two-phase parallel launcher for 04_run_simulations.py
#
# Rewritten from the previous version, which parallelized by splitting the
# WHOLE CIF library into N independent chunks, each running its own full
# Stage1->2->3 pipeline. That silently broke Stage 2's water-aware Pareto
# selection: selection needs to rank MOFs against each other across the
# WHOLE candidate pool, not N separate small pools ranked independently.
# It also pointed at a nonexistent RASPA install and the old output dir,
# and never passed --stage1-method/--gcmc-budget/--stage3-n-cycles at all.
#
# This version instead:
#   Phase A (single process, sequential): run Stage 1 (screening, cheap)
#     + Stage 2 (selection) ONCE globally over the whole CIF library,
#     producing gcmc_selected_mofs.txt -- the real "best N" MOF list.
#   Phase B (N parallel tmux workers): split ONLY the already-selected
#     MOFs across workers, each running Stage 3 only
#     (--skip-stage1 --skip-stage2, so no selection logic runs twice).
#     This is embarrassingly parallel and safe to split, since each
#     worker just runs independent RASPA jobs on its own MOF subset.
#
# Persistence: workers run under tmux + nohup, so they survive closing
# this terminal, VS Code, or an SSH/WSL session disconnect. They do NOT
# survive the Windows host actually sleeping, hibernating, or shutting
# down -- WSL2 is a VM under Windows, and Windows sleep suspends the VM
# (and everything in it) too. If you want a long run to survive your
# laptop's lid closing or auto-sleep, disable sleep on AC power from
# PowerShell (run once, as yourself, not from here):
#   powercfg /change standby-timeout-ac 0
#
# Resume-aware: re-running this script skips Phase A if
# gcmc_selected_mofs.txt already exists (pass --redo-selection to force
# it to rerun), and skips relaunching any worker that already has a
# completed adsorption_training.parquet from a previous run.
#
# Usage:
#   bash scripts/run_parallel_gcmc.sh [N_WORKERS]        # launch (or resume)
#   bash scripts/run_parallel_gcmc.sh --redo-selection [N_WORKERS]
#   bash scripts/run_parallel_gcmc.sh --status
#   bash scripts/run_parallel_gcmc.sh --kill
#   bash scripts/run_parallel_gcmc.sh --merge
# ═══════════════════════════════════════════════════════════════════════

PROJECT_ROOT="/home/fardo/uc_tpno_humid_pipeline/uc_tpno_humid_pipeline"
VENV="${PROJECT_ROOT}/thesis_env/bin/activate"
CIF_DIR="${PROJECT_ROOT}/data/intermediate/cifs_sanitized"
OUT_DIR="${PROJECT_ROOT}/data/processed/adsorption_v2"
CHUNK_DIR="${PROJECT_ROOT}/data/intermediate/cif_chunks_stage3"
LOG_DIR="${OUT_DIR}/logs"
RASPA_BIN="/home/fardo/RASPA2-install/bin/simulate"
RASPA_DATA="/home/fardo/RASPA2-install"
SCRIPT_PY="${PROJECT_ROOT}/scripts/04_run_simulations.py"
MERGE_PY="${PROJECT_ROOT}/scripts/merge_simulation_results.py"
PID_DIR="/tmp/gcmc_pids"
SELECTED_MOFS_FILE="${OUT_DIR}/gcmc_selected_mofs.txt"

# Stage 1 (screening) and Stage 3 (real training labels) settings --
# matches what we've been using: cheap screening, more careful final labels.
STAGE1_CYCLES=1000
STAGE3_CYCLES=4000
GCMC_BUDGET=500
MAX_MOFS=0
# FIX: "gctmmc" doesn't actually work with real RASPA2 -- verified by
# grepping RASPA2's own C source (input.c and everything else) for the
# keywords generate_tmmc_input() sends it (TMC_Method, TMC_CollectionMatrix,
# TMC_BiasUpdate): they appear NOWHERE. RASPA silently ignores them and
# runs a normal simulation that never writes a collection matrix, so every
# GC-TMMC job fails 100% of the time regardless of cycle count (confirmed
# in two separate real test runs). Falling back to plain GCMC, which is
# proven to work end-to-end (same code path validated by the water-model
# test in this session).
STAGE1_METHOD="gcmc"
TEMPERATURES="298.15,313.15,333.15"
PRESSURES="0.1,0.5,1.0,5.0,10.0"
# ADDED: select_mofs_for_gcmc() only ever reads Stage 1's lowest pressure
# point (Henry constant) and interpolates between min/max pressure
# (working capacity) at ONE temperature -- the rest of a full 3x5 sweep is
# wasted. A separate, smaller Stage 1 grid (1 temp x 3 pressures x 3
# components = 9 jobs/MOF) gets the cheap-screening benefit GC-TMMC would
# have given, using the plain GCMC method that actually works.
STAGE1_TEMPERATURES="313.15"
STAGE1_PRESSURES="0.1,1.0,10.0"
RHS="0.0,0.05,0.15"
DRY_CO2_FRAC=0.15

ACTION="${1:-launch}"

# ─────────────────────────────────────────────────────────────────────
# --status
# ─────────────────────────────────────────────────────────────────────
if [[ "$ACTION" == "--status" ]]; then
    echo "========================================"
    echo "STAGE 3 WORKER STATUS"
    echo "========================================"
    if [[ -f "${SELECTED_MOFS_FILE}" ]]; then
        # FIX: wc -l counts newline CHARACTERS. gcmc_selected_mofs.txt is
        # written via "\n".join(ids) with no trailing newline, so a 4-MOF
        # selection has only 3 newlines -- wc -l silently undercounted by
        # one (verified: reported 3 MOFs for an actual 4-MOF selection).
        n_selected=$(grep -c . "${SELECTED_MOFS_FILE}")
        echo "  Global selection: ${n_selected} MOFs (${SELECTED_MOFS_FILE})"
    else
        echo "  Global selection: NOT YET RUN (Phase A hasn't completed)"
    fi
    echo "----------------------------------------"
    running=0
    finished=0
    for pid_file in "${PID_DIR}"/worker_*.pid; do
        [[ -f "$pid_file" ]] || continue
        worker=$(basename "$pid_file" .pid)
        pid=$(cat "$pid_file")
        log="${LOG_DIR}/${worker}.log"
        done_rows=0
        worker_out="${OUT_DIR}/${worker}"
        if [[ -f "${worker_out}/adsorption_training.parquet" ]]; then
            done_rows="complete"
        elif [[ -f "$log" ]]; then
            # FIX: was anchored to the start of the line ("^\[..."), but the
            # actual log format is "TIME | LEVEL |   [job/total] MOF | T=...
            # P=... RH=...", so the "[" is never at position 0 -- this never
            # matched anything, always reporting progress~0 while real work
            # was happening (verified: log had real "[16/45] ... T=... P=...
            # RH=..." lines that this pattern silently missed).
            done_rows=$(grep -cE "\[[0-9]+/[0-9]+\].*T=.*P=.*RH=" "$log" 2>/dev/null || true)
        fi
        if kill -0 "$pid" 2>/dev/null; then
            echo "  ${worker}: PID=${pid} | RUNNING  | progress~${done_rows}"
            running=$((running+1))
        else
            echo "  ${worker}: PID=${pid} | STOPPED  | result=${done_rows}"
            finished=$((finished+1))
        fi
    done
    n_done=$(find "${OUT_DIR}" -maxdepth 2 -name "adsorption_training.parquet" -path "*/worker_*" 2>/dev/null | wc -l)
    echo "----------------------------------------"
    echo "  Running: ${running} | Stopped: ${finished}"
    echo "  Workers with completed output: ${n_done}"
    echo "========================================"
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────
# --kill
# ─────────────────────────────────────────────────────────────────────
if [[ "$ACTION" == "--kill" ]]; then
    echo "Killing all Stage 3 workers..."
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
    for s in $(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep '^gcmc_stage3_' || true); do
        tmux kill-session -t "$s" 2>/dev/null || true
    done
    echo "Done."
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────
# --merge
# ─────────────────────────────────────────────────────────────────────
if [[ "$ACTION" == "--merge" ]]; then
    source "${VENV}"
    python "${MERGE_PY}" \
        --worker-dir "${OUT_DIR}" \
        --output "${OUT_DIR}/adsorption_training.parquet"
    exit 0
fi

REDO_SELECTION=false
if [[ "$ACTION" == "--redo-selection" ]]; then
    REDO_SELECTION=true
    shift
    ACTION="${1:-launch}"
fi

# ─────────────────────────────────────────────────────────────────────
# Phase A: Stage 1 + Stage 2, ONCE, globally, sequentially.
# Skipped automatically on resume if the selection already exists.
# ─────────────────────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${PID_DIR}"

if [[ -f "${SELECTED_MOFS_FILE}" ]] && [[ "${REDO_SELECTION}" == "false" ]]; then
    echo "========================================"
    echo "PHASE A: SKIPPED (selection already exists)"
    echo "  ${SELECTED_MOFS_FILE}"
    echo "  Pass --redo-selection to force a fresh Stage 1+2 run."
    echo "========================================"
else
    echo "========================================"
    echo "PHASE A: Stage 1 (${STAGE1_METHOD}, ${STAGE1_CYCLES} cycles) + Stage 2"
    echo "  Screening all CIFs in ${CIF_DIR}"
    echo "========================================"
    source "${VENV}"
    python "${SCRIPT_PY}" \
        --cif-dir "${CIF_DIR}" \
        --output-dir "${OUT_DIR}" \
        --raspa-path "${RASPA_BIN}" \
        --raspa-data-dir "${RASPA_DATA}" \
        --max-mofs "${MAX_MOFS}" \
        --stage1-method "${STAGE1_METHOD}" \
        --gcmc-budget "${GCMC_BUDGET}" \
        --n-cycles "${STAGE1_CYCLES}" \
        --temperatures "${TEMPERATURES}" \
        --pressures "${PRESSURES}" \
        --stage1-temperatures "${STAGE1_TEMPERATURES}" \
        --stage1-pressures "${STAGE1_PRESSURES}" \
        --rhs "${RHS}" \
        --dry-co2-frac "${DRY_CO2_FRAC}" \
        --stage3-n-cycles "${STAGE3_CYCLES}" \
        --stop-after-stage2
        # FIX: Phase A never actually runs Stage 3 (--stop-after-stage2
        # exits first), so --stage3-n-cycles is a no-op here functionally.
        # But 04_run_simulations.py's startup banner prints it regardless,
        # and without this flag it silently defaulted to args.n_cycles
        # (Stage 1's cycle count) instead of the real Stage 3 value --
        # printing "Stage 3 cycles: 1000" during Phase A even though
        # Phase B's workers correctly use 4000. Passed through here purely
        # so the banner isn't misleading.
fi

if [[ ! -f "${SELECTED_MOFS_FILE}" ]]; then
    echo "ERROR: ${SELECTED_MOFS_FILE} still missing after Phase A. Check the log above."
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────
# Phase B: split the SELECTED MOFs (not the whole library) across
# N parallel tmux workers, each running Stage 3 only.
# ─────────────────────────────────────────────────────────────────────
if [[ -n "${1:-}" ]] && [[ "$1" =~ ^[0-9]+$ ]]; then
    N_WORKERS=$1
else
    N_CORES=$(nproc --all)
    N_WORKERS=$(( N_CORES > 1 ? N_CORES - 1 : 1 ))
    N_WORKERS=$(( N_WORKERS > 16 ? 16 : N_WORKERS ))
fi

mapfile -t SELECTED_IDS < "${SELECTED_MOFS_FILE}"
N_TOTAL=${#SELECTED_IDS[@]}
if [[ ${N_TOTAL} -eq 0 ]]; then
    echo "ERROR: No MOFs in ${SELECTED_MOFS_FILE}"
    exit 1
fi
if [[ ${N_WORKERS} -gt ${N_TOTAL} ]]; then
    N_WORKERS=${N_TOTAL}
fi
CHUNK_SIZE=$(( (N_TOTAL + N_WORKERS - 1) / N_WORKERS ))

echo "========================================"
echo "PHASE B: STAGE 3 PARALLEL LAUNCHER - ${N_WORKERS} workers"
echo "  Selected MOFs: ${N_TOTAL} | Chunk: ~${CHUNK_SIZE} each"
echo "  Stage 3 cycles: ${STAGE3_CYCLES}"
echo "========================================"

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
        mof_id="${SELECTED_IDS[$i]}"
        src="${CIF_DIR}/${mof_id}.cif"
        if [[ -f "${src}" ]]; then
            ln -sf "${src}" "${chunk_path}/${mof_id}.cif"
        else
            echo "  WARNING: selected MOF CIF not found: ${src}"
        fi
    done
    count=$(( end - start ))
    echo "  Worker ${w}: ${count} MOFs"
done

for (( w=0; w<N_WORKERS; w++ )); do
    worker="worker_${w}"
    WORKER_OUT="${OUT_DIR}/${worker}"

    # Resume: skip workers that already finished in a previous launch.
    if [[ -f "${WORKER_OUT}/adsorption_training.parquet" ]]; then
        echo "  ${worker}: already complete, skipping (delete its output to force a rerun)"
        continue
    fi

    SESSION="gcmc_stage3_${w}"
    CHUNK_PATH="${CHUNK_DIR}/worker_${w}"
    LOG_FILE="${LOG_DIR}/${worker}.log"
    PID_FILE="${PID_DIR}/${worker}.pid"
    mkdir -p "${WORKER_OUT}"

    tmux kill-session -t "${SESSION}" 2>/dev/null || true
    tmux new-session -d -s "${SESSION}"
    tmux send-keys -t "${SESSION}" "source ${VENV} && nohup python ${SCRIPT_PY} --cif-dir ${CHUNK_PATH} --output-dir ${WORKER_OUT} --raspa-path ${RASPA_BIN} --raspa-data-dir ${RASPA_DATA} --skip-stage1 --skip-stage2 --stage3-n-cycles ${STAGE3_CYCLES} --temperatures '${TEMPERATURES}' --pressures '${PRESSURES}' --rhs '${RHS}' --dry-co2-frac ${DRY_CO2_FRAC} > ${LOG_FILE} 2>&1 & echo \$! > ${PID_FILE} && echo ${worker} started PID: \$(cat ${PID_FILE})" ENTER
    echo "  Launched ${SESSION}"
done

sleep 3
echo ""
echo "All Stage 3 workers launched (or already complete)."
echo "  Status : bash scripts/run_parallel_gcmc.sh --status"
echo "  Kill   : bash scripts/run_parallel_gcmc.sh --kill"
echo "  Log    : tail -f ${LOG_DIR}/worker_0.log"
echo "  Merge  : bash scripts/run_parallel_gcmc.sh --merge"
echo ""
echo "Reminder: workers survive closing this terminal/SSH session, but NOT"
echo "the Windows host actually sleeping/hibernating. For a long unattended"
echo "run, disable sleep on AC power once from PowerShell (not from here):"
echo "  powercfg /change standby-timeout-ac 0"
