#!/bin/bash
# Run similarity search experiments with different error margins
# AIDS/PubChem/Chemical1M (tau=8): margin = 0, 1, 2, 4, 8, 16, 32, 64, 128
# SYN (tau=4): margin = 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1.0

SCRIPT="simple_search.py"
PYTHON="python"
LOG_DIR="../results/margin_experiments_logs"
mkdir -p "$LOG_DIR"

MARGINS="0 1 2 4 8 16 32 64 128"
SYN_MARGINS="0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 1.0"

echo "=============================================="
echo "Starting Error Margin Experiments"
echo "=============================================="
echo "Margins (AIDS/PubChem/Chemical1M): $MARGINS"
echo "Margins (SYN): $SYN_MARGINS"
echo ""

# AIDS: tau=8
echo "=============================================="
echo "Dataset: AIDS, tau=8"
echo "=============================================="
for margin in $MARGINS; do
    echo "  Running margin=$margin ..."
    $PYTHON $SCRIPT --dataset AIDS --tau 8 --error_margin $margin --query_start 0 --query_end 99 --use_parallel 2>&1 | tee "$LOG_DIR/AIDS_tau8_margin${margin}.log"
    echo "  Done."
done

# PubChem: tau=8
echo "=============================================="
echo "Dataset: PubChem, tau=8"
echo "=============================================="
for margin in $MARGINS; do
    echo "  Running margin=$margin ..."
    $PYTHON $SCRIPT --dataset PubChem --tau 8 --error_margin $margin --query_start 0 --query_end 99 --use_parallel 2>&1 | tee "$LOG_DIR/PubChem_tau8_margin${margin}.log"
    echo "  Done."
done

# SYN: tau=4 (uses different margin range)
echo "=============================================="
echo "Dataset: SYN, tau=4"
echo "=============================================="
for margin in $SYN_MARGINS; do
    echo "  Running margin=$margin ..."
    $PYTHON $SCRIPT --dataset SYN --tau 4 --error_margin $margin --query_start 0 --query_end 99 --use_parallel 2>&1 | tee "$LOG_DIR/SYN_tau4_margin${margin}.log"
    echo "  Done."
done

# Chemical1M: tau=8 (last because it takes very long)
echo "=============================================="
echo "Dataset: Chemical1M, tau=8"
echo "=============================================="
for margin in $MARGINS; do
    echo "  Running margin=$margin ..."
    $PYTHON $SCRIPT --dataset Chemical1M --tau 8 --error_margin $margin --query_start 0 --query_end 99 --use_parallel 2>&1 | tee "$LOG_DIR/Chemical1M_tau8_margin${margin}.log"
    echo "  Done."
done

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Results saved to: ../results/similarity_search/<dataset>/tau<X>_margin<Y>/"
echo "Logs saved to: $LOG_DIR/"
echo "=============================================="
