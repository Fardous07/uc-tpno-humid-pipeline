#!/usr/bin/env bash
# GCMC for validation Levels 3 and 4, with exactly the Stage 3 settings.
#
#   bash scripts/validation/validation_gcmc.sh prepare      # build the job list
#   bash scripts/validation/validation_gcmc.sh start [N]    # humid queue (N slots, default 11)
#                                                            # + Level 4 pure-gas isotherms (4 slots)
#   bash scripts/validation/validation_gcmc.sh status
#   bash scripts/validation/validation_gcmc.sh kill         # stops BOTH, and any RASPA job
#
# Humid runs (45 conditions = Stage 3 grid) for the 30 Level 3 MOFs
# (level3_mofs.txt, from select_level3_mofs.py) and the 4 Level 4 benchmark
# MOFs (level4_mofs.csv, from prepare_level4_cifs.py + assign_charges.py)
# go through scripts/stage3_queue.py pointed at data/processed/validation_gcmc.
# Level 4 also gets pure CO2 / N2 / H2O isotherms at 298.15 K for comparison
# with NIST ISODB experiments (output: validation_gcmc/level4_pure/<MOF>/).
set -euo pipefail

PROJECT_ROOT="/home/fardo/uc_tpno_humid_pipeline/uc_tpno_humid_pipeline"
PY="${PROJECT_ROOT}/thesis_env/bin/python"
QUEUE_PY="${PROJECT_ROOT}/scripts/stage3_queue.py"
VAL="${PROJECT_ROOT}/data/processed/validation_gcmc"
SRC_CIFS="${PROJECT_ROOT}/data/intermediate/validation_cifs"
L4_CIFS="${PROJECT_ROOT}/data/intermediate/validation_cifs_level4"
PURE_CIFS="${PROJECT_ROOT}/data/intermediate/validation_cifs_level4_pure"
RASPA_BIN="/home/fardo/RASPA2-install/bin/simulate"
RASPA_DATA="/home/fardo/RASPA2-install"
SESSION="gcmc_val"
PURE_SESSION="gcmc_val_pure"

# Level 4 pure-gas isotherms: 298.15 K is the most common experimental
# temperature in NIST ISODB for all four frameworks.
PURE_T="298.15"
PURE_P="0.01,0.05,0.1,0.2,0.5,1.0"
PURE_RH="0.05,0.1,0.2,0.4"
PURE_CYCLES=2000

ENV="QUEUE_OUT_DIR=${VAL} QUEUE_SELECTED=${VAL}/validation_mofs.txt \
QUEUE_SRC_CIF_DIR=${SRC_CIFS} QUEUE_CIF_DIR=${PROJECT_ROOT}/data/intermediate/cif_chunks_queue_val"

case "${1:-status}" in
    prepare)
        mkdir -p "${SRC_CIFS}" "${VAL}/logs"
        { cat "${VAL}/level3_mofs.txt"; tail -n +2 "${VAL}/level4_mofs.csv" | cut -d, -f2; } \
            | grep . > "${VAL}/validation_mofs.txt"
        while read -r m; do
            src="${PROJECT_ROOT}/data/intermediate/cifs_sanitized/${m}.cif"
            [[ -f "${src}" ]] || src="${L4_CIFS}/${m}.cif"
            ln -sfn "${src}" "${SRC_CIFS}/${m}.cif"
        done < "${VAL}/validation_mofs.txt"
        echo "$(wc -l < "${VAL}/validation_mofs.txt") MOFs -> ${VAL}/validation_mofs.txt"
        env ${ENV} "${PY}" "${QUEUE_PY}" prepare-fresh
        # One folder per Level 4 MOF so the four pure-gas runs go in parallel.
        tail -n +2 "${VAL}/level4_mofs.csv" | while IFS=, read -r name mof _; do
            mkdir -p "${PURE_CIFS}/${name}"
            ln -sfn "${L4_CIFS}/${mof}.cif" "${PURE_CIFS}/${name}/${mof}.cif"
        done
        ;;
    start)
        N="${2:-11}"
        if tmux has-session -t "${SESSION}" 2>/dev/null; then
            echo "Already running (tmux session ${SESSION})."; exit 1
        fi
        tmux new-session -d -s "${SESSION}" \
            "env ${ENV} setsid ${PY} ${QUEUE_PY} run ${N} >> ${VAL}/logs/queue.log 2>&1"
        tmux new-session -d -s "${PURE_SESSION}" "bash $(realpath "${BASH_SOURCE[0]}") _pure"
        sleep 3
        echo "Started: humid queue (${N} slots, tmux ${SESSION}) + 4 pure-gas runs (tmux ${PURE_SESSION})."
        echo "  Status : bash scripts/validation/validation_gcmc.sh status"
        ;;
    _pure)
        # Internal: the four Level 4 pure-gas runs in parallel (started by `start`).
        cd "${PROJECT_ROOT}"
        for d in "${PURE_CIFS}"/*/; do
            name=$(basename "${d}")
            "${PY}" scripts/04_run_simulations.py --cif-dir "${d}" \
                --output-dir "${VAL}/level4_pure/${name}" \
                --raspa-path "${RASPA_BIN}" --raspa-data-dir "${RASPA_DATA}" \
                --stage1-temperatures "${PURE_T}" --stage1-pressures "${PURE_P}" \
                --rhs "${PURE_RH}" --n-cycles "${PURE_CYCLES}" \
                --no-water-filter --stop-after-stage1 \
                > "${VAL}/logs/pure_${name}.log" 2>&1 &
        done
        wait
        ;;
    status)
        env ${ENV} "${PY}" "${QUEUE_PY}" status | sed 's/STAGE 3 QUEUE/VALIDATION QUEUE (Level 3+4)/'
        echo "Level 4 pure-gas isotherms (${PURE_T} K):"
        for d in "${PURE_CIFS}"/*/; do
            name=$(basename "${d}")
            ck="${VAL}/level4_pure/${name}/stage1_isotherms.checkpoint.csv"
            n=0; [[ -f "${ck}" ]] && n=$(( $(wc -l < "${ck}") - 1 ))
            done_flag=""; [[ -f "${VAL}/level4_pure/${name}/stage1_isotherms.parquet" ]] && done_flag="  DONE"
            echo "  ${name}: ${n}/16 jobs${done_flag}"
        done
        ;;
    kill)
        pkill -f "validation_gcmc.sh _pure" || true
        pkill -f "04_run_simulations.py --cif-dir ${PURE_CIFS}" || true
        env ${ENV} "${PY}" "${QUEUE_PY}" kill
        tmux kill-session -t "${SESSION}" 2>/dev/null || true
        tmux kill-session -t "${PURE_SESSION}" 2>/dev/null || true
        echo "Stopped. Resume with: bash scripts/validation/validation_gcmc.sh start"
        ;;
    *)
        echo "Usage: $0 {prepare|start [N]|status|kill}"; exit 1 ;;
esac
