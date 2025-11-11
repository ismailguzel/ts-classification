#!/bin/bash

# Load required module
module load apps/truba-ai/gpu-2024.0

### Model 1 (Binary Classification)
echo "=== Starting Model 1 Training ==="
cd /arf/home/iguzel/ts-stationary/hierarchical-ts-classification/03-models/hierarchical/model1_binary

# RAW mode
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
    --engine autogluon \
    --features-path ../../../data/features/unified-20k/allfeatures \
    --save-dir ./save_models/unified-20k/autogluon/allfeatures \
    --time-limit 3600 \
    --presets medium_quality_faster_train \
    2>&1 | tee ./out/autogluon_allfeatures-20k.out

# AutoGluon (selected features)
echo "Training Model 1 - AutoGluon Selected Features"
mkdir -p ./out
python -u autotrain_models1.py \
    --engine autogluon \
    --features-path ../../../data/features/unified-20k/selected \
    --save-dir ./save_models/unified-20k/autogluon/selected \
    --time-limit 3600 \
    --presets medium_quality_faster_train \
    2>&1 | tee ./out/autogluon_selected-20k.out

# PyCaret (all features)
echo "Training Model 1 - PyCaret All Features"
conda activate ts-pycaret
mkdir -p ./out
python -u autotrain_models1.py \
    --engine pycaret \
    --features-path ../../../data/features/unified-20k/allfeatures \
    --save-dir ./save_models/unified-20k/pycaret/allfeatures \
    2>&1 | tee ./out/pycaret_allfeatures-20k.out

# PyCaret (selected features)
echo "Training Model 1 - PyCaret Selected Features"
mkdir -p ./out
python -u autotrain_models1.py \
    --engine pycaret \
    --features-path ../../../data/features/unified-20k/selected \
    --save-dir ./save_models/unified-20k/pycaret/selected \
    2>&1 | tee ./out/pycaret_selected-20k.out

### Model 2 (5-Class Classification)
echo "=== Starting Model 2 Training ==="
cd /arf/home/iguzel/ts-stationary/hierarchical-ts-classification/03-models/hierarchical/model2_nonstationary

# RAW mode
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
    --engine autogluon \
    --features-path ../../../data/features/unified-20k/allfeatures \
    --save-dir ./save_models/unified-20k/autogluon/allfeatures \
    --time-limit 3600 \
    --presets medium_quality_faster_train \
    2>&1 | tee ./out/autogluon2_allfeatures-20k.out

# AutoGluon (selected features)
echo "Training Model 2 - AutoGluon Selected Features"
mkdir -p ./out
python -u autotrain_models2.py \
    --engine autogluon \
    --features-path ../../../data/features/unified-20k/selected \
    --save-dir ./save_models/unified-20k/autogluon/selected \
    --time-limit 3600 \
    --presets medium_quality_faster_train \
    2>&1 | tee ./out/autogluon2_selected-20k.out

# PyCaret (all features)
echo "Training Model 2 - PyCaret All Features"
conda activate ts-pycaret
mkdir -p ./out
python -u autotrain_models2.py \
    --engine pycaret \
    --features-path ../../../data/features/unified-20k/allfeatures \
    --save-dir ./save_models/unified-20k/pycaret/allfeatures \
    2>&1 | tee ./out/pycaret2_allfeatures-20k.out

# PyCaret (selected features)
echo "Training Model 2 - PyCaret Selected Features"
mkdir -p ./out
python -u autotrain_models2.py \
    --engine pycaret \
    --features-path ../../../data/features/unified-20k/selected \
    --save-dir ./save_models/unified-20k/pycaret/selected \
    2>&1 | tee ./out/pycaret2_selected-20k.out

echo "=== All training completed ==="
