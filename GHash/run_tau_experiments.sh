#!/bin/bash
# Run similarity search experiments with different tau values
# AIDS/PubChem: tau = 2, 4, 6, 8, 10, 12
# Chemical1M: tau = 2, 4, 6, 8, 10
# SYN: tau = 1, 2, 3, 4, 5, 6
# All with margin = 0.5

SCRIPT="simple_search.py"
PYTHON="python"
LOG_DIR="../results/tau_experiments_logs"
mkdir -p "$LOG_DIR"

MARGIN=0.5

echo "=============================================="
echo "Starting Tau Experiments (margin=$MARGIN)"
echo "=============================================="
echo ""

# AIDS: tau=2,4,6,8,10,12
echo "=============================================="
echo "Dataset: AIDS, tau=2,4,6,8,10,12"
echo "=============================================="
for tau in 2 4 6 8 10 12; do
    echo "  Running tau=$tau ..."
    $PYTHON $SCRIPT --dataset AIDS --tau $tau --error_margin $MARGIN --query_start 0 --query_end 99 --use_parallel 2>&1 | tee "$LOG_DIR/AIDS_tau${tau}_margin${MARGIN}.log"
    echo "  Done."
done

# PubChem: tau=2,4,6,8,10,12
echo "=============================================="
echo "Dataset: PubChem, tau=2,4,6,8,10,12"
echo "=============================================="
for tau in 2 4 6 8 10 12; do
    echo "  Running tau=$tau ..."
    $PYTHON $SCRIPT --dataset PubChem --tau $tau --error_margin $MARGIN --query_start 0 --query_end 99 --use_parallel 2>&1 | tee "$LOG_DIR/PubChem_tau${tau}_margin${MARGIN}.log"
    echo "  Done."
done

# SYN: tau=1,2,3,4,5,6
echo "=============================================="
echo "Dataset: SYN, tau=1,2,3,4,5,6"
echo "=============================================="
for tau in 1 2 3 4 5 6; do
    echo "  Running tau=$tau ..."
    $PYTHON $SCRIPT --dataset SYN --tau $tau --error_margin $MARGIN --query_start 0 --query_end 99 --use_parallel 2>&1 | tee "$LOG_DIR/SYN_tau${tau}_margin${MARGIN}.log"
    echo "  Done."
done

# Chemical1M: tau=2,4,6,8,10 (last because it takes very long)
echo "=============================================="
echo "Dataset: Chemical1M, tau=2,4,6,8,10"
echo "=============================================="
for tau in 2 4 6 8 10; do
    echo "  Running tau=$tau ..."
    $PYTHON $SCRIPT --dataset Chemical1M --tau $tau --error_margin $MARGIN --query_start 0 --query_end 99 --use_parallel 2>&1 | tee "$LOG_DIR/Chemical1M_tau${tau}_margin${MARGIN}.log"
    echo "  Done."
done

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Results saved to: ../results/similarity_search/<dataset>/tau<X>_margin${MARGIN}/"
echo "Logs saved to: $LOG_DIR/"
echo "=============================================="
