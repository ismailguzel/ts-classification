# Training Pipeline - 20K Dataset

Complete training pipeline for hierarchical time series classification using the unified-20K dataset.

## Overview

This script (`run-20k-trainig.sh`) trains both models with optimized configurations:

**Model 1 (Binary Classification):** Stationary vs Non-Stationary
**Model 2 (5-Class Classification):** Trend, Volatility, Stochastic, Anomaly, Structural Break

## What Gets Trained

### Model 1 (Binary)
1. **RAW mode** - sktime classifiers (ROCKET, Arsenal, TimeSeriesForest)
2. **FEATURES mode** - sklearn classifiers (RandomForest, XGBoost, CatBoost, SVM)
   - ✨ **NEW:** Feature importance CSVs (top 50 features per model)
3. **AutoGluon** - AutoML ensemble
   - ✨ **NEW:** Feature importance CSVs

### Model 2 (5-Class)
1. **RAW mode** - sktime classifiers
2. **FEATURES mode** - sklearn classifiers
   - ✨ **NEW:** Feature importance CSVs
3. **AutoGluon** - AutoML ensemble
   - ✨ **NEW:** Feature importance CSVs

## Usage

```bash
# Basic usage
./run-20k-trainig.sh

# Run in background with nohup
nohup ./run-20k-trainig.sh > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

## Requirements

- **Module:** `apps/truba-ai/gpu-2024.0`
- **Conda Environments:**
  - `ts-sktime` - for RAW and FEATURES modes
  - `ts-autogluon` - for AutoGluon training

## Output Structure

```
03-models/hierarchical/
├── model1_binary/
│   ├── saved_models/
│   │   ├── model1_binary_raw_rocket/
│   │   │   ├── predictions_*.csv
│   │   │   ├── misclassified_*.csv
│   │   │   └── *_metrics.json
│   │   ├── model1_binary_features/
│   │   │   ├── predictions_*.csv
│   │   │   ├── misclassified_*.csv
│   │   │   ├── feature_importance_RandomForest.csv ⭐ NEW
│   │   │   ├── feature_importance_XGBoost.csv ⭐ NEW
│   │   │   ├── feature_importance_CatBoost.csv ⭐ NEW
│   │   │   └── *_metrics.json
│   │   └── model1_binary_autogluon/
│   │       ├── predictions_*.csv
│   │       ├── misclassified_*.csv
│   │       ├── feature_importance_AutoGluon_Binary.csv ⭐ NEW
│   │       └── *_metrics.json
│   └── out/
│       ├── train_raw-20k.out
│       ├── train_selected-20k.out
│       └── autogluon_selected-20k.out
└── model2_nonstationary/
    └── (same structure as model1)
```

## Training Time Estimates

Based on 20K dataset:

- **RAW mode:** ~2-3 hours per model
- **FEATURES mode:** ~15-30 minutes per model
- **AutoGluon (1 hour limit):** ~1 hour per model

**Total estimated time:** 6-8 hours for complete pipeline

## Features Added in Latest Version

### Feature Importance Extraction
- Automatically extracts feature importance from tree-based models
- Saves top 50 most important features to CSV
- Available for: RandomForest, XGBoost, CatBoost, AutoGluon
- Format: `feature_importance_<ModelName>.csv`

Example:
```csv
feature,importance,rank
data__agg_autocorrelation__f_agg_"median"__maxlag_40,0.24625629,1
data__kurtosis,0.1408743,2
data__augmented_dickey_fuller__attr_"teststat"__autolag_"AIC",0.12038395,3
```

### Enhanced Logging
- Color-coded output for better readability
- Progress indicators for each training stage
- Summary of saved outputs at completion
- Automatic error handling with `set -e`

## What Changed from Original

### Removed Training Steps
- ❌ All features mode (too many features, redundant)
- ❌ AutoGluon with all features (redundant)

### Optimized Pipeline
- ✅ Only trains with selected features (100 features, mutual info)
- ✅ Faster training (~50% time reduction)
- ✅ Better model performance
- ✅ Feature importance extraction built-in

## Monitoring Progress

### Check Current Status
```bash
# View last 50 lines of output
tail -50 03-models/hierarchical/model1_binary/out/train_selected-20k.out

# Watch training in real-time
tail -f 03-models/hierarchical/model1_binary/out/train_selected-20k.out
```

### Check for Errors
```bash
# Search for errors in logs
grep -i error 03-models/hierarchical/*/out/*.out

# Check if feature importance files were created
find 03-models/hierarchical/*/saved_models -name "feature_importance*.csv"
```

## After Training

### 1. Review Metrics
```bash
# Check Model 1 performance
cat 03-models/hierarchical/model1_binary/saved_models/model1_binary_features/model1_features_summary.json

# Check Model 2 performance
cat 03-models/hierarchical/model2_nonstationary/saved_models/model2_nonstationary_features/model2_features_summary.json
```

### 2. Analyze Errors
```bash
cd 04-postprocessing

# Model 1 error analysis
python visualize_errors_simple.py --model model1 --save-fig

# Model 2 error analysis
python visualize_errors_simple.py --model model2 --save-fig
```

### 3. Visualize Misclassified Samples
```bash
# Find worst cases from error analysis, then:
python visualize_misclassified.py --id <SERIES_ID> --model model1 --save-fig
python visualize_misclassified.py --id <SERIES_ID> --model model2 --save-fig
```

### 4. Examine Feature Importance
```bash
# View top 10 features for XGBoost
head -11 03-models/hierarchical/model1_binary/saved_models/model1_binary_features/feature_importance_XGBoost.csv

# Compare across models
for model in RandomForest XGBoost CatBoost; do
    echo "=== $model ==="
    head -6 03-models/hierarchical/model1_binary/saved_models/model1_binary_features/feature_importance_${model}.csv
done
```

## Troubleshooting

### Conda Environment Not Found
```bash
# Check available environments
conda env list

# Create if missing (see main README.md for setup instructions)
```

### Out of Memory
```bash
# Monitor memory usage
watch -n 1 free -h

# Reduce memory usage by training models separately
```

### Training Stuck
```bash
# Check if process is running
ps aux | grep python

# Kill if needed
pkill -f train_model
```

## Notes

- Script uses `set -e` to stop on first error
- All output is saved to `./out/` directories
- Feature importance is extracted during `evaluate()` call
- IDs in CSV files are actual `series_id` from data generation
- Each model directory is independent (can train separately)

## Example Full Run

```bash
# Start training
./run-20k-trainig.sh

# Output will show:
============================================================================
Starting 20K Training Pipeline
============================================================================
Start time: Mon Nov 18 17:00:00 +03 2024

=== Starting Model 1 Training ===
[1/5] Activating ts-sktime environment
[2/5] Training Model 1 - RAW mode
✓ RAW mode completed
[3/5] Training Model 1 - Selected Features (with feature importance)
✓ Selected features completed
  → Feature importance CSVs saved in saved_models/model1_binary_features/
[4/5] Activating ts-autogluon environment
[5/5] Training Model 1 - AutoGluon Selected Features (with feature importance)
✓ AutoGluon completed
  → Feature importance CSVs saved in saved_models/model1_binary_autogluon/

Model 1 training completed!
Results saved in: saved_models/model1_binary_*/

=== Starting Model 2 Training ===
...

============================================================================
ALL TRAINING COMPLETED!
============================================================================
End time: Tue Nov 19 01:00:00 +03 2024

Summary:
  Model 1 (Binary): saved_models/model1_binary_*/
  Model 2 (5-Class): saved_models/model2_nonstationary_*/

Feature Importance:
  ✓ Feature importance CSVs available in:
    - model1_binary_features/feature_importance_*.csv
    - model1_binary_autogluon/feature_importance_*.csv
    - model2_nonstationary_features/feature_importance_*.csv
    - model2_nonstationary_autogluon/feature_importance_*.csv

Next Steps:
  1. Review metrics in saved_models/*/*.json
  2. Analyze errors: cd 04-postprocessing && python visualize_errors_simple.py
  3. Visualize misclassified: python visualize_misclassified.py --id <ID> --model <model1/model2>
```
