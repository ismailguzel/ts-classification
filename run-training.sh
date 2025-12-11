#!/bin/bash
# Training Pipeline for 20K Dataset
# Steps: Model 1 Training -> Model 2 Training

set -e  # Exit on error

# Load required module (if on cluster)
if command -v module &> /dev/null; then
    module load apps/truba-ai/gpu-2024.0
fi

# Activate environment
conda activate ts-sktime

# --- Configuration ---
SCALE="20k"

# Paths
BASE_DIR=$(pwd)
DATA_SELECTED="$BASE_DIR/data/features/unified-$SCALE/selected"

# Validate Input
if [ ! -d "$DATA_SELECTED" ]; then
    echo "✗ Error: Preprocessed data not found at $DATA_SELECTED"
    echo "  Please run 'run-preprocessing.sh' first."
    exit 1
fi

echo "============================================================"
echo "Starting Training Pipeline ($SCALE)"
echo "============================================================"
echo "Input: $DATA_SELECTED"
echo ""

# Model 1: Binary Classification
echo "Training Model 1 (Binary)..."
cd 03-models/hierarchical/model1_binary
mkdir -p ./output
python train_model1.py \
    --features-path "$DATA_SELECTED" \
    --save-dir ./output \
    --mode features

MODEL1_PATH="./output/model1_binary_features/model1_binary_classifier.pkl"
if [ ! -f "$MODEL1_PATH" ]; then
    echo "✗ Error: Model 1 training failed. Model file not found: $MODEL1_PATH"
    exit 1
fi

# Model 2: Non-Stationary Classification
echo "Training Model 2 (Non-Stationary)..."
cd ../model2_nonstationary
mkdir -p ./output
python train_model2.py \
    --features-path "$DATA_SELECTED" \
    --save-dir ./output \
    --mode features

MODEL2_PATH="./output/model2_nonstationary_features/model2_nonstationary_classifier.pkl"
if [ ! -f "$MODEL2_PATH" ]; then
    echo "✗ Error: Model 2 training failed. Model file not found: $MODEL2_PATH"
    exit 1
fi

echo "============================================================"
echo "Training Completed Successfully!"
echo "============================================================"
echo "Model 1 Output: 03-models/hierarchical/model1_binary/output/model1_binary_features"
echo "Model 2 Output: 03-models/hierarchical/model2_nonstationary/output/model2_nonstationary_features"
echo "============================================================"
