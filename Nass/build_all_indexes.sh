#!/bin/bash
# Build indexes for all datasets with multiple tau values.
# If a smaller tau results in CNB, larger taus are skipped.
# Usage: ./build_all_indexes.sh [-T hours]

TIMEOUT_FLAG=""
if [ "$1" = "-T" ] && [ -n "$2" ]; then
    TIMEOUT_FLAG="-T $2"
fi

NASS_INDEX="./nass-index"
DATA_DIR="../Gisma/datasets"

build_dataset() {
    local name=$1
    shift
    local taus=("$@")
    local db="$DATA_DIR/$name/db.txt"

    echo ""
    echo "========================================"
    echo "Dataset: $name"
    echo "Tau values: ${taus[*]}"
    echo "========================================"

    for tau in "${taus[@]}"; do
        echo ""
        echo "--- $name tau=$tau ---"
        $NASS_INDEX $tau "$db" $TIMEOUT_FLAG

        # check log for CNB
        local log="logs/$name/${name}_index_tau${tau}.log"
        if [ -f "$log" ] && grep -q "CNB" "$log"; then
            echo ""
            echo "*** $name tau=$tau: CNB — skipping remaining taus ***"
            return
        fi

        echo "--- $name tau=$tau: done ---"
    done
}

echo "============================================"
echo "Nass Index Build Script"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Timeout: ${TIMEOUT_FLAG:-24 hours (default)}"
echo "============================================"

build_dataset "AIDS"       2 4 6 8
build_dataset "PubChem"    2 4 6 8
build_dataset "SYN"        1 2

echo ""
echo "============================================"
echo "All done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
