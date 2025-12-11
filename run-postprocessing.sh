#!/bin/bash
# ============================================================================
# Post-Processing Pipeline
# ============================================================================
# Analysis and visualization of training results.
# Steps:
#   1. Visualize Feature Importance
#   2. Analyze Errors (Misclassifications)
#   3. Generate Report Figures
# ============================================================================

set -e  # Exit on error

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
echo "Start time: $(date)"
echo ""

# Create figures directory
mkdir -p "$FIGURES_DIR"

cd "$POST_DIR"

# Model paths (matching run-training.sh)
MODEL1_DIR="$BASE_DIR/03-models/hierarchical/model1_binary/output/model1_binary_features"
MODEL2_DIR="$BASE_DIR/03-models/hierarchical/model2_nonstationary/output/model2_nonstationary_features"

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

echo ""
echo "============================================================"
echo "Post-Processing Complete!"
echo "============================================================"
echo "Figures saved in: $FIGURES_DIR"
echo "End time: $(date)"
