#!/bin/bash
# Training Pipeline for 20K Dataset
# Steps: Model 1 Training -> Model 2 Training

set -e  # Exit on error

# --- Configuration ---
SCALE="5k"
FEATURE_TYPE="combined"  # Options: statistical, topological, combined

# Logging setup (skip if called from run.sh)
if [ -z "$CALLED_FROM_MASTER" ]; then
    LOG_DIR="logs/$SCALE"
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$LOG_DIR/training_${FEATURE_TYPE}_${TIMESTAMP}.log"
    mkdir -p "$LOG_DIR"
    
    # Redirect all output to log file and console
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

# Load required module (if on cluster)
if command -v module &> /dev/null; then
    module load apps/truba-ai/gpu-2024.0
fi

# Activate environment
conda activate ts-sktime

# Determine size suffix
if [ "$SCALE" == "5k" ]; then
    SIZE_SUFFIX="_5k"
else
    SIZE_SUFFIX=""
fi

# Paths
BASE_DIR=$(pwd)
DATA_SELECTED="$BASE_DIR/data/features/unified-$SCALE/${FEATURE_TYPE}_selected"

# Validate Input
if [ ! -d "$DATA_SELECTED" ]; then
    echo "✗ Error: Preprocessed data not found at $DATA_SELECTED"
    echo "  Please run 'run-preprocessing.sh' first."
    echo "  Or set FEATURE_TYPE to one of: statistical, topological, combined"
    exit 1
fi

echo "============================================================"
echo "Starting Training Pipeline ($SCALE - $FEATURE_TYPE features)"
echo "============================================================"
if [ -n "$LOG_FILE" ]; then
    echo "Log file: $LOG_FILE"
fi
echo "Input: $DATA_SELECTED"
echo "Output suffix: ${SIZE_SUFFIX:-none (20k default)}"
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

# Rename outputs based on scale and feature type
cd "$BASE_DIR"
if [ "$SCALE" == "5k" ]; then
    # 5k: Rename to include feature type and scale
    if [ -d "03-models/hierarchical/model1_binary/output/model1_binary_features" ]; then
        mv 03-models/hierarchical/model1_binary/output/model1_binary_features \
           03-models/hierarchical/model1_binary/output/model1_${FEATURE_TYPE}${SIZE_SUFFIX}
    fi
    
    if [ -d "03-models/hierarchical/model2_nonstationary/output/model2_nonstationary_features" ]; then
        mv 03-models/hierarchical/model2_nonstationary/output/model2_nonstationary_features \
           03-models/hierarchical/model2_nonstationary/output/model2_${FEATURE_TYPE}${SIZE_SUFFIX}
    fi
    
    OUTPUT1="03-models/hierarchical/model1_binary/output/model1_${FEATURE_TYPE}${SIZE_SUFFIX}"
    OUTPUT2="03-models/hierarchical/model2_nonstationary/output/model2_${FEATURE_TYPE}${SIZE_SUFFIX}"
else
    # 20k: Keep or rename based on feature type
    if [ "$FEATURE_TYPE" != "statistical" ]; then
        # Non-default feature type: rename
        if [ -d "03-models/hierarchical/model1_binary/output/model1_binary_features" ]; then
            mv 03-models/hierarchical/model1_binary/output/model1_binary_features \
               03-models/hierarchical/model1_binary/output/model1_${FEATURE_TYPE}
        fi
        
        if [ -d "03-models/hierarchical/model2_nonstationary/output/model2_nonstationary_features" ]; then
            mv 03-models/hierarchical/model2_nonstationary/output/model2_nonstationary_features \
               03-models/hierarchical/model2_nonstationary/output/model2_${FEATURE_TYPE}
        fi
        
        OUTPUT1="03-models/hierarchical/model1_binary/output/model1_${FEATURE_TYPE}"
        OUTPUT2="03-models/hierarchical/model2_nonstationary/output/model2_${FEATURE_TYPE}"
    else
        # Default: Keep original names
        OUTPUT1="03-models/hierarchical/model1_binary/output/model1_binary_features"
        OUTPUT2="03-models/hierarchical/model2_nonstationary/output/model2_nonstationary_features"
    fi
fi

echo "============================================================"
echo "Training Completed Successfully!"
echo "============================================================"
echo "Model 1 Output: $OUTPUT1"
echo "Model 2 Output: $OUTPUT2"
echo "============================================================"
