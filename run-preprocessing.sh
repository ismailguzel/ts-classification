#!/bin/bash
# Preprocessing Pipeline for 20K Dataset
# Steps: Feature Extraction -> Leakage Removal -> Feature Selection

set -e  # Exit on error

# Load required module (if on cluster)
if command -v module &> /dev/null; then
    module load apps/truba-ai/gpu-2024.0
fi

# Activate environment
conda activate ts-sktime

# --- Configuration ---
SCALE="20k"
N_JOBS=100
FEATURE_SET="efficient"
TOP_K_FEATURES=100

# Paths
BASE_DIR=$(pwd)
DATA_RAW="$BASE_DIR/data/raw/unified-$SCALE"
DATA_FEATURES="$BASE_DIR/data/features/unified-$SCALE/allfeatures"
DATA_SELECTED="$BASE_DIR/data/features/unified-$SCALE/selected"

# Validate Input
if [ ! -d "$DATA_RAW" ]; then
    echo "✗ Error: Input directory not found: $DATA_RAW"
    echo "  Please run 'run-generation.sh' first."
    exit 1
fi

# Create Output Directories
mkdir -p "$DATA_FEATURES"
mkdir -p "$DATA_SELECTED"

echo "============================================================"
echo "Starting Preprocessing Pipeline ($SCALE)"
echo "============================================================"
echo "Input:  $DATA_RAW"
echo "Output: $DATA_SELECTED"
echo ""

# 1. Feature Extraction
echo "Step 1: Feature Extraction (Chunked & Parallel)"
cd 02-preprocessing
python feature_extraction.py \
    --input "$DATA_RAW" \
    --output "$DATA_FEATURES" \
    --feature-set $FEATURE_SET \
    --n-jobs $N_JOBS

if [ ! -f "$DATA_FEATURES/features.parquet" ]; then
    echo "✗ Error: Feature extraction failed. Output file not found: $DATA_FEATURES/features.parquet"
    exit 1
fi

# 2. Leakage Removal
echo "Step 2: Removing Data Leakage Features"
python remove_leakage_features.py \
    --input "$DATA_FEATURES" \
    --no-backup

# 3. Feature Selection
echo "Step 3: Selecting Top $TOP_K_FEATURES Features"
python feature_selection.py \
    --input "$DATA_FEATURES" \
    --output "$DATA_SELECTED" \
    --n-features $TOP_K_FEATURES \
    --method mutual_info \
    --target all

if [ ! -d "$DATA_SELECTED" ] || [ -z "$(ls -A $DATA_SELECTED)" ]; then
    echo "✗ Error: Feature selection failed. Output directory empty: $DATA_SELECTED"
    exit 1
fi

echo "============================================================"
echo "Preprocessing Complete!"
echo "Output Directory: $DATA_SELECTED"
echo "============================================================"
