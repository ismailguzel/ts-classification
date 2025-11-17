#!/bin/bash
# ============================================================================
# Hierarchical Time Series Classification - 20K Training Pipeline
# ============================================================================
# This script trains all models (Model 1 & Model 2) with different modes:
#   - RAW mode: Uses raw time series with sktime classifiers
#   - FEATURES mode: Uses TSFresh-extracted features with sklearn classifiers
#   - AutoGluon: AutoML with extracted features
#
# Output Structure:
#   saved_models/
#   ├── model1_binary_raw_<classifier>/      # RAW mode outputs
#   ├── model1_binary_features/              # FEATURES mode outputs
#   ├── model1_binary_autogluon/             # AutoGluon outputs
#   ├── model2_nonstationary_raw_<classifier>/
#   ├── model2_nonstationary_features/
#   └── model2_nonstationary_autogluon/
# ============================================================================

# Load required module
module load apps/truba-ai/gpu-2024.0

### Model 1 (Binary Classification)
echo "=== Starting Model 1 Training ==="
cd /arf/home/iguzel/ts-stationary/hierarchical-ts-classification/03-models/hierarchical/model1_binary

# RAW mode (trains all classifiers: rocket, arsenal, tsforest)
echo "Training Model 1 - RAW mode"
conda activate ts-sktime
mkdir -p ./out
python -u train_model1.py \
    --mode raw \
    --data-path ../../../data/raw/unified-20k \
    2>&1 | tee ./out/train_raw-20k.out

# FEATURES mode (all features)
echo "Training Model 1 - All Features"
mkdir -p ./out
python -u train_model1.py \
    --mode features \
    --features-path ../../../data/features/unified-20k/allfeatures \
    2>&1 | tee ./out/train_allfeatures-20k.out

# FEATURES mode (selected features)
echo "Training Model 1 - Selected Features"
mkdir -p ./out
python -u train_model1.py \
    --mode features \
    --features-path ../../../data/features/unified-20k/selected \
    2>&1 | tee ./out/train_selected-20k.out

# AutoGluon (all features)
echo "Training Model 1 - AutoGluon All Features"
conda activate ts-autogluon
mkdir -p ./out
python -u autotrain_models1.py \
    --features-path ../../../data/features/unified-20k/allfeatures \
    --time-limit 3600 \
    --presets medium_quality_faster_train \
    2>&1 | tee ./out/autogluon_allfeatures-20k.out

# AutoGluon (selected features)
echo "Training Model 1 - AutoGluon Selected Features"
mkdir -p ./out
python -u autotrain_models1.py \
    --features-path ../../../data/features/unified-20k/selected \
    --time-limit 3600 \
    --presets medium_quality_faster_train \
    2>&1 | tee ./out/autogluon_selected-20k.out

### Model 2 (5-Class Classification)
echo "=== Starting Model 2 Training ==="
cd /arf/home/iguzel/ts-stationary/hierarchical-ts-classification/03-models/hierarchical/model2_nonstationary

# RAW mode (trains all classifiers: rocket, arsenal, tsforest)
echo "Training Model 2 - RAW mode"
conda activate ts-sktime
mkdir -p ./out
python -u train_model2.py \
    --mode raw \
    --data-path ../../../data/raw/unified-20k \
    2>&1 | tee ./out/train2_raw-20k.out

# FEATURES mode (all features)
echo "Training Model 2 - All Features"
mkdir -p ./out
python -u train_model2.py \
    --mode features \
    --features-path ../../../data/features/unified-20k/allfeatures \
    2>&1 | tee ./out/train2_allfeatures-20k.out

# FEATURES mode (selected features)
echo "Training Model 2 - Selected Features"
mkdir -p ./out
python -u train_model2.py \
    --mode features \
    --features-path ../../../data/features/unified-20k/selected \
    2>&1 | tee ./out/train2_selected-20k.out

# AutoGluon (all features)
echo "Training Model 2 - AutoGluon All Features"
conda activate ts-autogluon
mkdir -p ./out
python -u autotrain_models2.py \
    --features-path ../../../data/features/unified-20k/allfeatures \
    --time-limit 3600 \
    --presets medium_quality_faster_train \
    2>&1 | tee ./out/autogluon2_allfeatures-20k.out

# AutoGluon (selected features)
echo "Training Model 2 - AutoGluon Selected Features"
mkdir -p ./out
python -u autotrain_models2.py \
    --features-path ../../../data/features/unified-20k/selected \
    --time-limit 3600 \
    --presets medium_quality_faster_train \
    2>&1 | tee ./out/autogluon2_selected-20k.out

echo "=== All training completed ==="
