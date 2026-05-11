#!/bin/bash
# ============================================================================
# Data Generation Pipeline (betise)
# ============================================================================
# Generates synthetic time series dataset (config-driven).
# Output: data/raw/dataset/dataset.parquet
#
# Usage:
#   bash run-generation.sh           # full dataset (1000/class, 39 classes)
#   bash run-generation.sh test      # test dataset  (100/class, 39 classes)
#   bash run-generation.sh topo-test # topology-friendly dataset (100/class, 10 classes)
# ============================================================================

set -e

PYTHON="${PYTHON:-python}"
MODE=${1:-full}

if [ "$MODE" == "test" ]; then
    CONFIG="test-config.json"
    OUTPUT_FILE="data/raw/test/test.parquet"
elif [ "$MODE" == "topo-test" ]; then
    CONFIG="topo-test-config.json"
    OUTPUT_FILE="data/raw/topo-test/topo-test.parquet"
else
    CONFIG="full-dataset-config.json"
    OUTPUT_FILE="data/raw/dataset/dataset.parquet"
fi

echo "============================================================"
echo "Data Generation — $MODE ($CONFIG)"
echo "============================================================"
echo "Start time: $(date)"

cd 01-data-generation
$PYTHON generate.py --config "$CONFIG"
cd ..

if [ -f "$OUTPUT_FILE" ]; then
    echo "✓ Done: $OUTPUT_FILE"
else
    echo "✗ Error: Output file not found: $OUTPUT_FILE"
    exit 1
fi

echo "End time: $(date)"
