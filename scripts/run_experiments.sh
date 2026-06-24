#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh
# Automate running all experiments for Phase 1 and Phase 2.
#
# Phase 1: 5 models (naive, snaive, sarimax, lstm, prophet)
# Phase 2: 5 models (naive/snaive reuse Phase 1 logic but save to phase2/)
#
# Usage:
#   bash scripts/run_experiments.sh           # default uses REAL data
#   bash scripts/run_experiments.sh --mock    # use mock data for testing
# =============================================================================

set -e

MOCK_FLAG=""
if [[ "$1" == "--mock" ]]; then
    MOCK_FLAG="--mock"
    echo "[INFO] Using MOCK data (generated)"
else
    echo "[INFO] Using REAL data (parquet files)"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo ""
echo "============================================================"
echo "  PROJECT ROOT: $PROJECT_ROOT"
echo "============================================================"
echo ""

# -----------------------------------------------------------------
# Phase 1: all 5 models
# -----------------------------------------------------------------
echo "============================================================"
echo "  [PHASE 1] Running 5 models: naive, snaive, sarimax, lstm, prophet"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

python "$SCRIPT_DIR/run_experiments.py" \
    --phase 1 \
    --models naive snaive sarimax lstm prophet \
    $MOCK_FLAG

echo ""
echo "============================================================"
echo "  [PHASE 1] Done!"
echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# -----------------------------------------------------------------
# Phase 2: all 5 models (naive/snaive produce same results as P1
# but are saved to phase2/ for completeness)
# -----------------------------------------------------------------
echo "============================================================"
echo "  [PHASE 2] Running 5 models: naive, snaive, sarimax, lstm, prophet"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

python "$SCRIPT_DIR/run_experiments.py" \
    --phase 2 \
    --models naive snaive sarimax lstm prophet \
    $MOCK_FLAG

echo ""
echo "============================================================"
echo "  [PHASE 2] Done!"
echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# -----------------------------------------------------------------
# Summary
# -----------------------------------------------------------------
echo ""
echo "============================================================"
echo "  [DONE] All experiments completed."
echo "  Results: $PROJECT_ROOT/results/"
echo "  Metrics: $PROJECT_ROOT/results/metrics_comparison.csv"
echo "  Time:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""
