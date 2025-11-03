# 02-preprocessing

Data preprocessing and feature engineering pipeline.

## Overview

⚠️ Note: Preprocessing is optional. Modeller RAW (sktime) veya FEATURES (sklearn) modlarında çalıştırılabilir.

This directory contains scripts for:
1. **Feature Extraction**: Extract time series features using TSFresh
2. **Feature Selection**: Select relevant features for classification

---

## 📂 Structure

```
02-preprocessing/
├── extract_features.py        # TSFresh feature extraction
├── feature_selection.py       # Feature selection methods
└── README.md                   # This file
```

---

## 🚀 Quick Start

### 1. Extract Features with TSFresh

```bash
# Extract features with efficient settings
python extract_features.py \
    --input ../data/raw/unified-90k \
    --output ../data/features \
    --feature-set efficient \
    --n-jobs 4
```

Options:
- `--feature-set`: `minimal`, `efficient`, `comprehensive`
- `--n-jobs`: Number of parallel processes

Output:
- `../data/features/features.parquet`
- `../data/features/labels.parquet`
- `../data/features/feature_names.txt`

### 2. Feature Selection

```bash
# Select features using mutual information
python feature_selection.py \
    --input ../data/features \
    --output ../data/features/selected \
    --method mutual_info \
    --n-features 100 \
    --target all
```

Methods:
- `variance`: remove low variance and correlated features
- `correlation`: remove highly correlated features
- `statistical`: ANOVA F-test based selection
- `mutual_info`: mutual information based selection
- `importance`: Random Forest feature importance

Targets:
- `binary`: stationary vs non-stationary
- `primary`: primary category
- `sub`: sub-category
- `all`: all targets

Output:
- Legacy (kept for reference):
  - `features_binary_mutual_info.parquet`, `features_primary_mutual_info.parquet`, `features_sub_mutual_info.parquet`
  - `feature_names_*.txt`
- Standardized per-target directories (for training):
  - `../data/features/selected/binary/{features.parquet, labels.parquet}`
  - `../data/features/selected/primary/{features.parquet, labels.parquet}`
  - `../data/features/selected/sub/{features.parquet, labels.parquet}`

---

## Feature Engineering with TSFresh

TSFresh extracts hundreds of time series features including:

Statistical features
- Mean, median, standard deviation
- Quantiles, min, max, range
- Skewness, kurtosis

Temporal features
- Autocorrelation (various lags)
- Partial autocorrelation
- Trend strength
- Seasonality

Frequency domain
- FFT coefficients
- Power spectral density
- Spectral entropy

Complexity
- Approximate entropy
- Sample entropy
- Lempel-Ziv complexity

Change detection
- Mean/variance changes
- Linear trends
- Number of peaks

---

## Configuration

Feature extraction settings

Edit feature sets in `extract_features.py`:

```python
# Minimal (fast, ~20 features)
from tsfresh.feature_extraction import MinimalFCParameters
settings = MinimalFCParameters()

# Efficient (balanced, ~200 features)
from tsfresh.feature_extraction import EfficientFCParameters
settings = EfficientFCParameters()

# Comprehensive (slow, ~800 features)
from tsfresh.feature_extraction import ComprehensiveFCParameters
settings = ComprehensiveFCParameters()
```

Feature selection thresholds

Adjust thresholds in `feature_selection.py`:

```python
# Variance threshold (remove low variance features)
variance_threshold = 0.01  # Default

# Correlation threshold (remove correlated features)
correlation_threshold = 0.95  # Default

# Number of features to select
n_features = 100  # Default
```

---

<!-- Expected performance/results section removed to keep README usage-focused. -->

---

## Dependencies

Install required packages:

```bash
pip install tsfresh scikit-learn pandas numpy
```

Or use requirements:

```bash
pip install -r ../requirements.txt
```

---

## Tips

1. Prefer `efficient` settings for a balanced feature set
2. Use feature selection to reduce dimensionality
3. Parallelize with `--n-jobs`
4. Use chunked processing to manage memory
5. Inspect feature importance for insights

---

## Notes

- TSFresh handles missing values with imputation steps
- Scaling/normalization is not applied here by default (apply in modeling if needed)
- Selected features are specific to target choice
- You can customize feature sets in the script

---

## Troubleshooting

Out of memory?
- Reduce `--n-jobs` parameter
- Use `minimal` feature set
- Process data in smaller batches

Throughput issues?
- Increase `--n-jobs` (up to CPU cores)
- Use `minimal` or `efficient` settings
- Consider sampling for initial experiments

Poor feature quality?
- Try different feature selection methods
- Increase `--n-features` parameter
- Use `comprehensive` settings for more features

---

## 📚 References

- TSFresh Documentation: https://tsfresh.readthedocs.io/
- Feature Engineering Guide: https://tsfresh.readthedocs.io/en/latest/text/feature_extraction.html
- Feature Selection: https://tsfresh.readthedocs.io/en/latest/text/feature_selection.html


python extract_features.py --input ../data/raw/unified-test --output ../data/features/unified-test/allfeatures


python feature_selection.py --input ../data/features/unified-test/allfeatures --output ../data/features/unified-test/selected --method mutual_info --n-features 100 --target all


# Train on all extracted features (no selection)
python train_model1.py --mode features --features-path ../../../data/features/unified-test/allfeatures


# Train on selected binary features (recommended for Model 1)
python train_model1.py --mode features --features-path ../../../data/features/unified-test/selected/binary