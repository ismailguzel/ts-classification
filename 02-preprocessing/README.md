# 02-preprocessing

Data preprocessing and feature engineering pipeline.

## Overview

 Note: Preprocessing is optional. Modeller RAW (sktime) veya FEATURES (sklearn) modlarında çalıştırılabilir.

This directory contains scripts for:
1. **Feature Extraction**: Extract time series features using TSFresh
2. **Feature Selection**: Select relevant features for classification

---

##  Structure

```
02-preprocessing/
├── extract_dask.py            # TSFresh feature extraction (Dask parallel, RECOMMENDED)
├── feature_selection.py       # Feature selection methods
├── topological-features/      # Experimental: Topological data analysis (TDA)
│   ├── topological_features.py
│   ├── test_topological.py
│   ├── topological_demo.ipynb
│   └── README.md
└── README.md                   # This file
```

**Note:** `extract_features.py` has been deprecated. Use `extract_dask.py` for all feature extraction (faster, more efficient).

**Experimental**: `topological-features/` contains research code for TDA-based features (not integrated in main pipeline).

---

##  Quick Start

### 1. Extract Features with TSFresh (extract_dask.py)

**Feature Set Options:**
- `minimal`: ~780 features (**RECOMMENDED**, default - fast, good coverage)
- `efficient`: optimized subset for classification
- `comprehensive`: ~1200+ features (slow, high memory, for research)

#### Basic Usage

```bash
# Quick start with minimal features (recommended)
python extract_dask.py \
    --input ../data/raw/unified-5k \
    --output ../data/features/unified-5k/minimal \
    --feature-set minimal \
    --n-workers 4 \
    --threads-per-worker 2

# Large dataset (20K series)
python extract_dask.py \
    --input ../data/raw/unified-20k \
    --output ../data/features/unified-20k/minimal \
    --feature-set minimal \
    --n-workers 55 \
    --threads-per-worker 1 \
    --memory-limit 0

# Comprehensive features (for research/comparison)
python extract_dask.py \
    --input ../data/raw/unified-20k \
    --output ../data/features/unified-20k/comprehensive \
    --feature-set comprehensive \
    --n-workers 55
```

#### Advanced Options

```bash
# With external Dask scheduler
python extract_dask.py \
    --input ../data/raw/unified-20k \
    --output ../data/features/unified-20k/minimal \
    --scheduler-address tcp://localhost:8786

# Smoke test (limited files)
python extract_dask.py \
    --input ../data/raw/unified-20k \
    --output ../data/features/test \
    --max-files 20
```

**Command-line Options:**
- `--input`: Input directory with parquet files
- `--output`: Output directory for features
- `--feature-set`: `minimal` (default), `efficient`, or `comprehensive`
- `--n-workers`: Number of Dask workers
- `--threads-per-worker`: Threads per worker
- `--memory-limit`: Memory limit per worker (0 = unlimited)
- `--scheduler-address`: External Dask scheduler (optional)
- `--max-files`: Limit files for testing
- `--no-client`: Don't create Dask client (use default scheduler)

**Output Files:**
- `features.parquet` - Extracted features
- `labels.parquet` - Classification labels
- `feature_names.txt` - List of all feature names

## Feature Selection

### Available Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `mutual_info` | Mutual Information | Non-linear relationships |
| `statistical` | ANOVA F-test | Linear relationships |
| `importance` | Random Forest | Feature importance ranking |
| `correlation` | Correlation-based | Removing redundant features |
| `variance` | Low variance removal | Preprocessing |

### Usage

```bash
# Default: unified-5k dataset
python feature_selection.py --method mutual_info --n-features 100 --target all

# For specific datasets
python feature_selection.py \
    --input ../data/features/unified-test/allfeatures \
    --output ../data/features/unified-test/selected \
    --method mutual_info \
    --n-features 100 \
    --target all

# Larger datasets (20K example with logging)
python -u feature_selection.py \
    --input ../data/features/unified-20k/allfeatures \
    --output ../data/features/unified-20k/selected \
    --method mutual_info \
    --n-features 100 \
    --target all 2>&1 | tee selection-20k.out
```


### Target Options

- `all`: Select features for all tasks (binary + primary + sub)
- `binary`: Binary classification only (stationary vs non-stationary)
- `primary`: Primary category classification (5 classes)
- `sub`: Sub-category classification (detailed types)

### Output Structure

Feature selection creates standardized directories:

```
data/features/unified-5k/selected/
├── binary/
│   ├── features.parquet       # Selected features for binary task
│   ├── labels.parquet          # Binary labels
│   └── feature_names.txt       # List of selected feature names
├── primary/
│   ├── features.parquet        # Selected features for primary task
│   ├── labels.parquet          # Primary category labels
│   └── feature_names.txt
└── sub/
    ├── features.parquet
    ├── labels.parquet
    └── feature_names.txt
```

Legacy files (kept for backward compatibility):
- `features_binary_mutual_info.parquet`
- `features_primary_mutual_info.parquet`
- `features_sub_mutual_info.parquet`

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

##  References

- TSFresh Documentation: https://tsfresh.readthedocs.io/
- Feature Engineering Guide: https://tsfresh.readthedocs.io/en/latest/text/feature_extraction.html
- Feature Selection: https://tsfresh.readthedocs.io/en/latest/text/feature_selection.html


python extract_features.py --input ../data/raw/unified-test --output ../data/features/unified-test/allfeatures


python feature_selection.py --input ../data/features/unified-test/allfeatures --output ../data/features/unified-test/selected --method mutual_info --n-features 100 --target all


# Train on all extracted features (no selection)
python train_model1.py --mode features --features-path ../../../data/features/unified-test/allfeatures


# Train on selected binary features (recommended for Model 1)
python train_model1.py --mode features --features-path ../../../data/features/unified-test/selected/binary