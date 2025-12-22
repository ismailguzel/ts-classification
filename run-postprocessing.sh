#!/bin/bash
# ============================================================================
# Post-Processing Pipeline
# ============================================================================
# Analysis and visualization of training results.
# Steps:
#   1. Visualize Feature Importance
#   2. Analyze Errors (Misclassifications)
#   3. Generate Report Figures
#
# Usage:
#   bash run-postprocessing.sh [FEATURE_TYPE] [DATASET_SIZE]
#
# Arguments:
#   FEATURE_TYPE  - Feature type: statistical/topological/combined (default: statistical)
#   DATASET_SIZE  - Dataset size: 5k/20k (default: 20k)
#
# Examples:
#   bash run-postprocessing.sh statistical 20k
#   bash run-postprocessing.sh topological 5k
#   bash run-postprocessing.sh combined 5k
# ============================================================================

set -e  # Exit on error

# Parse arguments
FEATURE_TYPE=${1:-statistical}
DATASET_SIZE=${2:-20k}

# Load required module (if on cluster)
if command -v module &> /dev/null; then
    module load apps/truba-ai/gpu-2024.0
fi

# Activate environment
conda activate ts-sktime

# --- Configuration ---
BASE_DIR=$(pwd)
POST_DIR="$BASE_DIR/04-postprocessing"
FIGURES_DIR="$POST_DIR/figures"

# Determine suffix for dataset size
if [ "$DATASET_SIZE" == "5k" ]; then
    SIZE_SUFFIX="_5k"
else
    SIZE_SUFFIX=""
fi

# Validate Environment
if [ ! -d "$POST_DIR" ]; then
    echo "✗ Error: Post-processing directory not found: $POST_DIR"
    exit 1
fi

# Check if models exist (basic check)
MODEL1_DIR="$BASE_DIR/03-models/hierarchical/model1_binary/output"
MODEL2_DIR="$BASE_DIR/03-models/hierarchical/model2_nonstationary/output"

if [ ! -d "$MODEL1_DIR" ] && [ ! -d "$MODEL2_DIR" ]; then
    echo "⚠ Warning: Model output directories not found. Scripts might fail if models aren't trained."
    echo "  Model 1: $MODEL1_DIR"
    echo "  Model 2: $MODEL2_DIR"
fi

echo "============================================================"
echo "Starting Post-Processing Pipeline"
echo "============================================================"
echo "Feature Type: $FEATURE_TYPE"
echo "Dataset Size: $DATASET_SIZE"
echo "Start time: $(date)"
echo ""

# Create figures directory with feature type subdirectory
FIGURES_DIR="$FIGURES_DIR/${FEATURE_TYPE}${SIZE_SUFFIX}"
mkdir -p "$FIGURES_DIR"

cd "$POST_DIR"

# Model paths (matching run-training.sh)
MODEL1_DIR="$BASE_DIR/03-models/hierarchical/model1_binary/output/model1_${FEATURE_TYPE}${SIZE_SUFFIX}"
MODEL2_DIR="$BASE_DIR/03-models/hierarchical/model2_nonstationary/output/model2_${FEATURE_TYPE}${SIZE_SUFFIX}"

# 1. Feature Importance Analysis
echo "Step 1: Analyzing Feature Importance..."

# Model 1 (Binary)
echo "  - Model 1 (Binary)..."
python visualize_feature_importance.py \
    --model model1 \
    --saved-models-dir "$MODEL1_DIR" \
    --top 30 \
    --save-fig \
    --normalize sum

# Model 2 (5-Class)
echo "  - Model 2 (5-Class)..."
python visualize_feature_importance.py \
    --model model2 \
    --saved-models-dir "$MODEL2_DIR" \
    --top 30 \
    --save-fig \
    --normalize sum

# 2. Error Analysis
echo "Step 2: Analyzing Errors..."

# Model 1 Errors
echo "  - Model 1 Errors..."
python visualize_errors_simple.py \
    --model model1 \
    --saved-models-dir "$MODEL1_DIR" \
    --save-fig

# Model 2 Errors
echo "  - Model 2 Errors..."
python visualize_errors_simple.py \
    --model model2 \
    --saved-models-dir "$MODEL2_DIR" \
    --save-fig

cd "$BASE_DIR"

cd "$BASE_DIR"

echo ""
echo "============================================================"
echo "Post-Processing Complete!"
echo "============================================================"
echo "Feature Type: $FEATURE_TYPE"
echo "Dataset Size: $DATASET_SIZE"
echo "Figures saved in: $FIGURES_DIR"
echo "End time: $(date)"
