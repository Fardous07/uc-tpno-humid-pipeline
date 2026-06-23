#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# 01_download_data.sh — Download raw MOF data for UC-TPNO pipeline
#
# Data sources:
#   1. CoRE MOF 2019 (ASR/FSR subsets)
#   2. ARC-MOF database
#   3. MOFX-DB (DFT-optimised structures)
#   4. Boyd & Woo WS24 (water stability dataset)
#   5. NIST ISODB (experimental isotherms)
#
# Usage:
#   bash scripts/01_download_data.sh [--data-dir data/raw]
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

DATA_DIR="${1:-data/raw}"
mkdir -p "$DATA_DIR"

echo "=============================================="
echo " UC-TPNO Data Download"
echo " Target: $DATA_DIR"
echo "=============================================="

# ── 1. CoRE MOF 2019 ────────────────────────────────────
echo ""
echo "[1/5] CoRE MOF 2019..."
CORE_DIR="$DATA_DIR/core_mof_2019"
mkdir -p "$CORE_DIR"
if [ ! -f "$CORE_DIR/README" ]; then
    echo "  → Download from https://zenodo.org/record/3370144"
    echo "  → Place ASR and FSR zip files in $CORE_DIR"
    echo "  → (Automated download requires zenodo API token)"
else
    echo "  → Already present."
fi

# ── 2. ARC-MOF ──────────────────────────────────────────
echo ""
echo "[2/5] ARC-MOF..."
ARC_DIR="$DATA_DIR/arc_mof"
mkdir -p "$ARC_DIR"
if [ ! -f "$ARC_DIR/README" ]; then
    echo "  → Download from https://zenodo.org/record/7481842"
    echo "  → Place CIF archive in $ARC_DIR"
else
    echo "  → Already present."
fi

# ── 3. MOFX-DB ──────────────────────────────────────────
echo ""
echo "[3/5] MOFX-DB..."
MOFX_DIR="$DATA_DIR/mofx_db"
mkdir -p "$MOFX_DIR"
echo "  → Access at https://mof.tech.northwestern.edu/databases"
echo "  → Download DFT-optimised CIFs into $MOFX_DIR"

# ── 4. WS24 water stability ─────────────────────────────
echo ""
echo "[4/5] WS24 water stability..."
WS24_DIR="$DATA_DIR/ws24"
mkdir -p "$WS24_DIR"
echo "  → Download from DOI: 10.1021/acs.chemmater.3c02296"
echo "  → Place CSV files in $WS24_DIR"

# ── 5. NIST ISODB ───────────────────────────────────────
echo ""
echo "[5/5] NIST ISODB experimental data..."
NIST_DIR="$DATA_DIR/nist_isodb"
mkdir -p "$NIST_DIR"
if command -v git &>/dev/null; then
    if [ ! -d "$NIST_DIR/isodb-library-main" ]; then
        echo "  → Cloning ISODB library..."
        git clone --depth 1 https://github.com/NIST-ISODB/isodb-library.git \
            "$NIST_DIR/isodb-library-main" 2>/dev/null || \
            echo "  → Git clone failed. Download manually."
    else
        echo "  → Already cloned."
    fi
else
    echo "  → git not available. Download from https://github.com/NIST-ISODB/isodb-library"
fi

echo ""
echo "=============================================="
echo " Download step complete."
echo " Next: python scripts/02_preprocess_all.py"
echo "=============================================="