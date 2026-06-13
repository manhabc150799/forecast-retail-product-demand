#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh
# Script tu dong hoa chay toan bo experiments cho ca Phase 1 va Phase 2.
#
# Luong 1 (Phase 1): chay 5 models (naive, snaive, sarimax, lstm, prophet)
# Luong 2 (Phase 2): chi chay 3 models co enhanced features (sarimax, lstm, prophet)
#
# Su dung:
#   bash scripts/run_experiments.sh           # mac dinh dung MOCK data
#   bash scripts/run_experiments.sh --real    # dung real parquet data
# =============================================================================

set -e  # dung ngay khi co bat ky lenh nao fail

# truyen --real neu muon dung data that
REAL_FLAG=""
if [[ "$1" == "--real" ]]; then
    REAL_FLAG="--real"
    echo "[INFO] Su dung REAL data (parquet files)"
else
    echo "[INFO] Su dung MOCK data (generated)"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo ""
echo "============================================================"
echo "  PROJECT ROOT: $PROJECT_ROOT"
echo "============================================================"
echo ""

# -----------------------------------------------------------------
# Luong 1: Phase 1 - tat ca 5 models
# -----------------------------------------------------------------
echo "============================================================"
echo "  [PHASE 1] Bat dau chay 5 models: naive, snaive, sarimax, lstm, prophet"
echo "  Thoi gian bat dau: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

python "$SCRIPT_DIR/run_experiments.py" \
    --phase 1 \
    --models naive snaive sarimax lstm prophet \
    $REAL_FLAG

echo ""
echo "============================================================"
echo "  [PHASE 1] Hoan thanh!"
echo "  Thoi gian ket thuc: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# -----------------------------------------------------------------
# Luong 2: Phase 2 - 3 models co enhanced features
# Theo guideline, chi SARIMAX, LSTM va Prophet duoc chay voi Phase 2
# Naive va SNaive khong dung exogenous nen ket qua giong Phase 1
# -----------------------------------------------------------------
echo "============================================================"
echo "  [PHASE 2] Bat dau chay 3 models: sarimax, lstm, prophet"
echo "  Thoi gian bat dau: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

python "$SCRIPT_DIR/run_experiments.py" \
    --phase 2 \
    --models sarimax lstm prophet \
    $REAL_FLAG

echo ""
echo "============================================================"
echo "  [PHASE 2] Hoan thanh!"
echo "  Thoi gian ket thuc: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# -----------------------------------------------------------------
# Tong ket
# -----------------------------------------------------------------
echo ""
echo "============================================================"
echo "  [DONE] Tat ca experiments da chay xong."
echo "  Ket qua luu tai: $PROJECT_ROOT/results/"
echo "  File so sanh    : $PROJECT_ROOT/results/metrics_comparison.csv"
echo "  Thoi gian       : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""
