#!/bin/bash
# GHash Similarity Search Script
# Run this script to get results for the comparison table

# Configuration
PYTHON="python"  # Change to your python path if needed, e.g., ~/anaconda3/envs/graph_hashing/bin/python
SCRIPT="simple_search.py"

# Output directory for results
RESULTS_DIR="../results/similarity_search"
mkdir -p "$RESULTS_DIR"

# Log file
LOG_FILE="$RESULTS_DIR/ghash_similarity_search_$(date +%Y%m%d_%H%M%S).log"

echo "GHash Similarity Search" | tee "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# AIDS dataset: tau = 2, 4, 6, 8, 10, 12
echo "" | tee -a "$LOG_FILE"
echo "=== AIDS Dataset ===" | tee -a "$LOG_FILE"
for tau in 2 4 6 8 10 12; do
    echo "Running AIDS tau=$tau ..." | tee -a "$LOG_FILE"
    $PYTHON $SCRIPT --dataset AIDS --tau $tau --query_start 0 --query_end 99 --use_parallel 2>&1 | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

# PubChem dataset: tau = 2, 4, 6, 8, 10, 12
echo "" | tee -a "$LOG_FILE"
echo "=== PubChem Dataset ===" | tee -a "$LOG_FILE"
for tau in 2 4 6 8 10 12; do
    echo "Running PubChem tau=$tau ..." | tee -a "$LOG_FILE"
    $PYTHON $SCRIPT --dataset PubChem --tau $tau --query_start 0 --query_end 99 --use_parallel 2>&1 | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

# Chemical1M dataset: tau = 2, 4, 6, 8, 10
echo "" | tee -a "$LOG_FILE"
echo "=== Chemical1M Dataset ===" | tee -a "$LOG_FILE"
for tau in 2 4 6 8 10; do
    echo "Running Chemical1M tau=$tau ..." | tee -a "$LOG_FILE"
    $PYTHON $SCRIPT --dataset Chemical1M --tau $tau --query_start 0 --query_end 99 --use_parallel 2>&1 | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

# SYN dataset: tau = 1, 2, 3, 4, 5, 6
echo "" | tee -a "$LOG_FILE"
echo "=== SYN Dataset ===" | tee -a "$LOG_FILE"
for tau in 1 2 3 4 5 6; do
    echo "Running SYN tau=$tau ..." | tee -a "$LOG_FILE"
    $PYTHON $SCRIPT --dataset SYN --tau $tau --query_start 0 --query_end 99 --use_parallel 2>&1 | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

echo "========================================" | tee -a "$LOG_FILE"
echo "Completed at: $(date)" | tee -a "$LOG_FILE"
echo "Results saved to: $LOG_FILE" | tee -a "$LOG_FILE"
