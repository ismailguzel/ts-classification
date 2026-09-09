#!/bin/bash
# ============================================================================
# Data Generation Pipeline (betise)
# ============================================================================
# Generates a synthetic time series dataset (config-driven).
# Output: data/raw/<mode>/<mode>.parquet
#
# Usage:
#   bash run-generation.sh shape             # Study A,  9 classes, 900 series
# ============================================================================

set -e

PYTHON="${PYTHON:-python}"
MODE=${1:-shape}

BASE_DIR=$(pwd)
source "$BASE_DIR/modes.sh"

CONFIG="$CONFIG_NAME"
OUTPUT_FILE="$RAW_PARQUET"

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
