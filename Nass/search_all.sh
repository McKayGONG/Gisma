#!/bin/bash
# Run searches for all datasets and tau values using pre-built indexes.
# Usage: ./search_all.sh

NASS="./nass"
DATA_DIR="../Gisma/datasets"

search_dataset() {
    local name=$1
    shift
    local taus=("$@")
    local db="$DATA_DIR/$name/db.txt"
    local queries="$DATA_DIR/$name/queries.txt"

    echo ""
    echo "========================================"
    echo "Dataset: $name"
    echo "Tau values: ${taus[*]}"
    echo "========================================"

    for tau in "${taus[@]}"; do
        local idx="indexes/$name/${name}_tau${tau}.idx"

        echo ""
        echo "--- $name tau=$tau ---"

        if [ ! -f "$idx" ]; then
            echo "*** Index $idx not found, skipping ***"
            continue
        fi

        $NASS $tau "$db" "$idx" -q "$queries"
        echo "--- $name tau=$tau: done ---"
    done
}

echo "============================================"
echo "Nass Search Script"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

search_dataset "AIDS"    2 4 6 8
search_dataset "PubChem" 2 4 6 8
search_dataset "SYN"     1 2

echo ""
echo "============================================"
echo "All done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
