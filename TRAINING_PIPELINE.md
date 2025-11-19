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

### Training Outputs

**Model Files:**
- `model*_classifier.pkl` - Trained model (RAW/FEATURES modes)
- `model*_features_summary.json` - Comprehensive evaluation metrics
- `predictions_*.csv` - All test predictions with true labels
- `misclassified_*.csv` - Incorrectly classified samples
- `feature_importance_*.csv` - Top 50 features (FEATURES/AutoGluon modes)

**Organized by Mode:**
- `model1_binary_raw_*` - sktime classifiers
- `model1_binary_features` - sklearn classifiers with feature importance
- `model1_binary_autogluon` - AutoML ensemble with feature importance
- `model2_nonstationary_*` - Same structure for 5-class model

**Logs:**
- Training outputs saved to `out/*.out` files

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
# View last 50 lines of training output
tail -50 03-models/hierarchical/model1_binary/out/train_selected-20k.out

# Watch training in real-time
tail -f 03-models/hierarchical/model1_binary/out/train_selected-20k.out
```

### Check for Errors
```bash
# Search for errors in logs
grep -i error 03-models/hierarchical/*/out/*.out

# Verify feature importance files were created
find 03-models/hierarchical/*/saved_models -name "feature_importance*.csv" | wc -l
# Expected: 8 files (4 for Model 1, 4 for Model 2)
```

## After Training

### 1. Generate Visualizations
```bash
# Run automated post-processing pipeline
./run-postprocessing.sh
```

**Output**: 6 publication-quality figures
- Error analysis for both models (confusion matrices, misclassification patterns)
- Feature importance visualizations (top-20 comparisons, category breakdowns)
- Detailed logs saved to `04-postprocessing/visualization.out`

### 2. Review Results
```bash
# Check detailed metrics
cat 03-models/hierarchical/model1_binary/saved_models/model1_binary_features/model1_features_summary.json

# View generated figures
ls -lh 04-postprocessing/figures/

# Read comprehensive analysis
less 04-postprocessing/visualization.out
```

### 3. Examine Specific Features
```bash
# View top 10 features for XGBoost
head -11 03-models/hierarchical/model1_binary/saved_models/model1_binary_features/feature_importance_XGBoost.csv

# Compare feature importance across models
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
- All training logs saved to `out/` directories
- Feature importance automatically extracted for tree-based models
- IDs in CSV files are actual `series_id` from data generation
- Each model directory is independent (can train separately)
- RAW mode trains sktime classifiers (ROCKET, Arsenal)
- FEATURES mode trains sklearn classifiers (RF, XGBoost, CatBoost, SVM)
- AutoGluon mode trains ensemble with automated hyperparameter tuning

## Complete Workflow Example

```bash
# Step 1: Generate data (if not already done)
cd 01-data-generation
python generate.py --scale 20k

# Step 2: Extract and select features (if not already done)
cd ../02-preprocessing
python extract_dask.py --input ../data/raw/unified-20k --output ../data/features/unified-20k/allfeatures
python feature_selection.py --input ../data/features/unified-20k/allfeatures --output ../data/features/unified-20k/selected

# Step 3: Train all models
cd ..
./run-20k-trainig.sh

# Step 4: Generate visualizations
./run-postprocessing.sh

# Step 5: Review results
cat TECHNICAL_REPORT.md
open 04-postprocessing/figures/*.png
```

## Next Steps After Training

1. **Review Performance**: Check `TECHNICAL_REPORT.md` for comprehensive analysis
2. **Examine Figures**: View `04-postprocessing/figures/` for error analysis and feature importance
3. **Analyze Logs**: Read `04-postprocessing/visualization.out` for detailed statistics
4. **Compare Models**: Review JSON metrics in `saved_models/*/model*_summary.json`
5. **Investigate Errors**: Examine `misclassified_*.csv` files for error patterns
