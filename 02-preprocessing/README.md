# 02-preprocessing

Data preprocessing and feature engineering pipeline.

## Overview

This directory contains scripts for:
1. **Feature Extraction**: Extract time series features using TSFresh
2. **Feature Selection**: Select relevant features for classification
3. **Data Transformation**: Normalize, scale, and prepare data for models

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

Extract comprehensive time series features:

```bash
# Extract features with efficient settings (recommended)
python extract_features.py \
    --input ../data/raw/unified-150k \
    --output ../data/features \
    --feature-set efficient \
    --n-jobs 4
```

**Options:**
- `--feature-set`: Choose from `minimal`, `efficient`, `comprehensive`
  - `minimal`: ~20 features, fast (~30 min)
  - `efficient`: ~200 features, balanced (~2 hours) ⭐ **Recommended**
  - `comprehensive`: ~800 features, slow (~4-6 hours)
- `--n-jobs`: Number of parallel processes (default: 4)

**Output:**
- `../data/features/features.parquet`: Extracted features
- `../data/features/labels.parquet`: Original labels
- `../data/features/feature_names.txt`: List of feature names

### 2. Feature Selection

Select most relevant features for each classification task:

```bash
# Select features using mutual information (recommended)
python feature_selection.py \
    --input ../data/features \
    --output ../data/features/selected \
    --method mutual_info \
    --n-features 100 \
    --target all
```

**Methods:**
- `variance`: Remove low variance and correlated features
- `correlation`: Remove highly correlated features
- `statistical`: ANOVA F-test based selection
- `mutual_info`: Mutual information based selection ⭐ **Recommended**
- `importance`: Random Forest feature importance

**Targets:**
- `binary`: Stationary vs non-stationary
- `primary`: Primary category (AR, MA, ARMA, etc.)
- `sub`: Sub-category (specific types)
- `all`: All targets

**Output:**
- `features_binary_mutual_info.parquet`: Selected features for binary classification
- `features_primary_mutual_info.parquet`: Selected features for primary classification
- `features_sub_mutual_info.parquet`: Selected features for sub-category classification
- `feature_names_*.txt`: Lists of selected feature names

---

## 📊 Feature Engineering with TSFresh

TSFresh extracts hundreds of time series features including:

### Statistical Features
- Mean, median, standard deviation
- Quantiles, min, max, range
- Skewness, kurtosis

### Temporal Features
- Autocorrelation (various lags)
- Partial autocorrelation
- Trend strength
- Seasonality

### Frequency Domain
- FFT coefficients
- Power spectral density
- Spectral entropy

### Complexity
- Approximate entropy
- Sample entropy
- Lempel-Ziv complexity

### Change Detection
- Mean/variance changes
- Linear trends
- Number of peaks

---

## ⚙️ Configuration

### Feature Extraction Settings

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

### Feature Selection Thresholds

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

## 📈 Expected Results

### Feature Extraction
- **Time**: 2-4 hours (efficient settings, 150K series)
- **Size**: ~5-8 GB feature matrix
- **Features**: ~200 features per series

### Feature Selection
- **Time**: 30-60 minutes
- **Features**: 50-150 selected features
- **Improvement**: Faster training, better generalization

---

## 🔧 Dependencies

Install required packages:

```bash
pip install tsfresh scikit-learn pandas numpy
```

Or use requirements:

```bash
pip install -r ../requirements.txt
```

---

## 💡 Tips

1. **Start with efficient settings**: Balance between speed and performance
2. **Use feature selection**: Reduces overfitting and speeds up training
3. **Parallel processing**: Use `--n-jobs` to speed up extraction
4. **Memory management**: Process in batches if running out of memory
5. **Feature importance**: Analyze which features are most important for your task

---

## 📝 Notes

- TSFresh automatically handles missing values
- Features are normalized/scaled during extraction
- Selected features are specific to each classification task
- Can extract custom features by modifying settings

---

## 🆘 Troubleshooting

**Out of memory?**
- Reduce `--n-jobs` parameter
- Use `minimal` feature set
- Process data in smaller batches

**Too slow?**
- Increase `--n-jobs` (up to CPU cores)
- Use `minimal` or `efficient` settings
- Consider sampling for initial experiments

**Poor feature quality?**
- Try different feature selection methods
- Increase `--n-features` parameter
- Use `comprehensive` settings for more features

---

## 📚 References

- TSFresh Documentation: https://tsfresh.readthedocs.io/
- Feature Engineering Guide: https://tsfresh.readthedocs.io/en/latest/text/feature_extraction.html
- Feature Selection: https://tsfresh.readthedocs.io/en/latest/text/feature_selection.html
