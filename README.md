# Hierarchical Time Series Classification

A pipeline for hierarchical classification of time series data using TSFresh feature engineering and machine learning models.

## Project Overview

This project implements a hierarchical classification system for time series data:

```
Input Time Series
    ↓
[Model 1: Binary]  →  Stationary (0) ✓
    ↓
  Non-Stationary (1)
    ↓
[Model 2: 5-Class]  →  Trend (0)
              →  Volatility (1)
              →  Stochastic (2)
              →  Anomaly (3)
              →  Structural Break (4)
```

### Key Features
- Synthetic time series generation (scalable: 1.5K to 200K samples)
- Three training modes: RAW (sktime), FEATURES (sklearn), AutoTrain (AutoGluon)
- Hierarchical two-stage classification
- Comprehensive evaluation metrics with CSV exports

---

##  Project Structure

```
hierarchical-ts-classification/
├── 01-data-generation/          # Data generation
│   ├── generate.py              # Scalable dataset generation (5K-200K)
│   ├── verify_ids.py            # Data integrity verification
│   ├── config.py                # Dataset configuration (11 scales)
│   └── jobs-slurm/              # Optional SLURM job scripts
│
├── 02-preprocessing/            # Feature engineering (OPTIONAL)
│   ├── extract_dask.py          # TSFresh feature extraction (Dask parallel)
│   ├── feature_selection.py    # Feature selection methods
│   ├── topological-features/    # Experimental: TDA-based features
│   └── README.md                # Preprocessing docs
│
├── 03-models/
│   ├── hierarchical/            # Hierarchical models
│   │   ├── model1_binary/       # Level 1: Binary (Stat vs Non-stat)
│   │   │   ├── train_model1.py      # Dual-mode training (RAW/FEATURES)
│   │   │   ├── autotrain_models1.py # AutoML training (AutoGluon)
│   │   │   ├── test_model1.py       # Testing script
│   │   │   └── README.md            # Documentation
│   │   │
│   │   ├── model2_nonstationary/    # Level 2: 5-Class (Non-stat types)
│   │   │   ├── train_model2.py      # Dual-mode training (RAW/FEATURES)
│   │   │   ├── autotrain_models2.py # AutoML training (AutoGluon)
│   │   │   ├── test_model2.py       # Testing script
│   │   │   └── README.md            # Documentation
│   │   │
│   │   └── README.md            # Models overview
│   │
│   └── utils/                   # Shared utilities
│       ├── metrics.py           # ModelEvaluator class
│       ├── data_utils.py        # Data loading utilities
│       └── constants.py         # Shared constants
│
├── data/                        # Data storage
│   ├── raw/                     # Raw time series
│   ├── features/                # TSFresh features (optional)
│   └── README.md                # Data documentation
│
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── smoke_test.sh                # Quick validation test
├── run-20k-trainig.sh           # Batch training script
└── TRAINING_REPORT_20K.md       # Training results report
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/ismailguzel/hierarchical-ts-classification.git
cd hierarchical-ts-classification
pip install -r requirements.txt
```

### 2. Generate Data

```bash
cd 01-data-generation

# Generate test dataset (recommended for quick testing)
python generate.py --scale test  # 1.5K samples

# Generate larger datasets (strategic progression)
python generate.py --scale 5k    # 5K samples
python generate.py --scale 10k   # 10K samples
python generate.py --scale 20k   # 20K samples
python generate.py --scale 90k   # 90K samples (baseline)
python generate.py --scale 200k  # 200K samples

# See all available scales
python config.py
```

### 3. Train Models

```bash
# Model 1: Binary Classification
cd ../03-models/hierarchical/model1_binary
python train_model1.py --mode raw --classifier rocket

# Model 2: 5-Class Classification
cd ../model2_nonstationary
python train_model2.py --mode raw --classifier rocket
```

### 4. Test Models

```bash
# Test with default models
cd 03-models/hierarchical/model1_binary
python test_model1.py

cd ../model2_nonstationary
python test_model2.py

# Test specific models
python test_model1.py --model-path saved_models/model1_binary_raw_rocket/model1_binary_classifier.pkl
python test_model2.py --model-path saved_models/model2_nonstationary_raw_arsenal/model2_nonstationary_classifier.pkl
```

### Alternative Workflows

**FEATURES Mode** (TSFresh + sklearn):
```bash
cd 02-preprocessing
python extract_dask.py \
    --input ../data/raw/unified-5k \
    --output ../data/features/unified-5k/allfeatures \
    --n-workers 4
python feature_selection.py \
    --input ../data/features/unified-5k/allfeatures \
    --output ../data/features/unified-5k/selected \
    --method mutual_info \
    --target all
cd ../03-models/hierarchical/model1_binary
python train_model1.py --mode features
```

**AutoTrain Mode** (AutoGluon):
```bash
# After feature extraction and selection
cd 03-models/hierarchical/model1_binary

# Using AutoGluon
python autotrain_models1.py \
    --features-path ../../../data/features/unified-5k/selected \
    --time-limit 3600 \
    --presets medium_quality_faster_train

# Model 2 AutoTrain
cd ../model2_nonstationary
python autotrain_models2.py \
    --features-path ../../../data/features/unified-5k/selected \
    --time-limit 3600 \
    --presets medium_quality_faster_train
```

**Note:** AutoTrain requires feature extraction first (see 02-preprocessing/).

**Smoke Test** (Quick Validation):
```bash
# Tests all modes with small dataset
bash smoke_test.sh
```

**Batch Training** (20K Dataset):
```bash
# Runs complete training pipeline
bash run-20k-trainig.sh
```

---

## Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DATA GENERATION (01-data-generation/)                       │
│    • generate.py → Scalable datasets (test/5k/10k/.../200k)   │
│    • config.py → 11 scale presets (1.5K to 200K)              │
│    • verify_ids.py → Data integrity check                      │
└─────────────────────────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. PREPROCESSING (02-preprocessing/) [OPTIONAL]                │
│    • extract_dask.py → TSFresh features (Dask parallel)        │
│    • feature_selection.py → Select relevant features           │
│    • topological-features/ → Experimental TDA features          │
│    └─→ Only for FEATURES/AutoTrain modes                       │
└─────────────────────────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. MODEL TRAINING (03-models/hierarchical/)                    │
│                                                                 │
│    A. RAW Mode (Recommended)                                   │
│       → Direct time series → sktime classifiers                │
│                                                                 │
│    B. FEATURES Mode                                            │
│       → TSFresh features → sklearn classifiers                 │
│                                                                 │
│    C. AutoTrain Mode                                           │
│       → TSFresh features → AutoML (AutoGluon)                  │
└─────────────────────────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. HIERARCHICAL CLASSIFICATION                                 │
│                                                                 │
│    Model 1 (Binary)                                            │
│    ├─→ Stationary (0) ✓                                        │
│    └─→ Non-Stationary (1)                                      │
│            ↓                                                    │
│    Model 2 (5-Class)                                           │
│    ├─→ Trend (0)                                               │
│    ├─→ Volatility (1)                                          │
│    ├─→ Stochastic (2)                                          │
│    ├─→ Anomaly (3)                                             │
│    └─→ Structural Break (4)                                    │
└─────────────────────────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. TESTING & EVALUATION                                        │
│    • test_model1.py / test_model2.py                           │
│    • Comprehensive metrics (JSON + CSV exports)                │
│    • Predictions and misclassified samples analysis            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Training Modes

| Mode | Input | Classifiers | When to Use |
|------|-------|-------------|-------------|
| **RAW** | Time series | sktime (ROCKET, Arsenal) | Quick start, baseline |
| **FEATURES** | TSFresh features | sklearn (XGBoost, SVM) | Feature analysis, interpretability |
| **AutoTrain** | TSFresh features | AutoML (AutoGluon) | Automated tuning |

See individual README files for detailed configuration options.

---

## Output Structure

Training scripts automatically organize outputs into mode-specific directories:

```
03-models/hierarchical/model1_binary/saved_models/
├── model1_binary_raw_rocket/
│   ├── model1_binary_classifier.pkl
│   ├── metrics.json
│   ├── predictions.csv
│   └── misclassified.csv
│
├── model1_binary_features/
│   ├── model1_binary_classifier.pkl
│   ├── metrics.json
│   ├── predictions.csv
│   └── misclassified.csv
│
└── model1_binary_autogluon/
    ├── models/              # AutoGluon models
    ├── metadata.pkl         # AutoGluon metadata
    ├── metrics.json
    ├── predictions.csv
    └── misclassified.csv
```

**Output Files:**
- `model*_classifier.pkl` - Trained model (RAW/FEATURES modes)
- `metrics.json` - Comprehensive evaluation metrics
- `predictions.csv` - All predictions with true labels
- `misclassified.csv` - Incorrectly classified samples
- `models/` - AutoGluon ensemble models directory
- `metadata.pkl` - AutoGluon training metadata

---

## Requirements

```bash
pip install -r requirements.txt

# Optional: AutoGluon support
pip install autogluon.tabular
```

Core dependencies:
- `tsfresh` - Time series feature extraction
- `scikit-learn` - Traditional ML classifiers
- `sktime` - Time series classifiers (ROCKET, Arsenal)
- `pandas`, `numpy` - Data manipulation
- `dask` - Parallel feature extraction

---

## Utilities

The `03-models/utils/` module provides shared functionality:

- **`metrics.py`**: `ModelEvaluator` class for comprehensive model evaluation
  - Automatic predictions CSV export
  - Misclassified samples CSV export
  - JSON metrics with nested structure
  
- **`data_utils.py`**: Data loading and preprocessing utilities
  - RAW mode: Time series loading
  - FEATURES mode: Parquet file handling
  
- **`constants.py`**: Shared constants and label mappings

---

## Documentation

Detailed documentation in subdirectories:

- **Data Generation**: [`01-data-generation/README.md`](01-data-generation/README.md)
- **Preprocessing**: [`02-preprocessing/README.md`](02-preprocessing/README.md)
- **Model Training**: [`03-models/hierarchical/README.md`](03-models/hierarchical/README.md)
- **Model 1 Details**: [`03-models/hierarchical/model1_binary/README.md`](03-models/hierarchical/model1_binary/README.md)
- **Model 2 Details**: [`03-models/hierarchical/model2_nonstationary/README.md`](03-models/hierarchical/model2_nonstationary/README.md)

---

##  Example Commands (20K Dataset)

### Data Generation
```bash
python generate.py --scale 20k
# Output: /arf/home/iguzel/ts-stationary/hierarchical-ts-classification/01-data-generation/generation-20k.out
```

### Feature Extraction and Selection
```bash
cd 02-preprocessing

# Step 1: Extract features from raw time series (creates ~800 TSFresh features)
python extract_dask.py \
    --input ../data/raw/unified-20k \
    --output ../data/features/unified-20k/allfeatures \
    --n-workers 55 \
    --memory-limit 0

# Output: allfeatures/features.parquet, allfeatures/labels.parquet

# Step 2: Select top features for each classification task
python feature_selection.py \
    --input ../data/features/unified-20k/allfeatures \
    --output ../data/features/unified-20k/selected \
    --method mutual_info \
    --n-features 100 \
    --target all

# Output: 
#   selected/binary/features.parquet + labels.parquet    (for Model 1)
#   selected/primary/features.parquet + labels.parquet   (for Model 2)
```

**Why separate feature selection for binary and primary?**
- Different tasks optimize different features via mutual information
- Binary (Model 1): Features that distinguish stationary vs non-stationary
- Primary (Model 2): Features that distinguish 5 non-stationary types
- Models automatically load from correct subdirectory (binary/ or primary/)


### Model 1 (Binary Classification)
```bash

conda activate ts-sktime
# RAW mode
python -u train_model1.py \
    --mode raw \
    --data-path ../../../data/raw/unified-20k \
    2>&1 | tee train_raw-20k.out

# FEATURES mode (all features)
python -u train_model1.py \
    --mode features \
    --features-path ../../../data/features/unified-20k/allfeatures \
    2>&1 | tee train_allfeatures-20k.out

# FEATURES mode (selected features)
python -u train_model1.py \
    --mode features \
    --features-path ../../../data/features/unified-20k/selected \
    2>&1 | tee train_selected-20k.out

conda activate ts-autogluon
# AutoGluon (all features)
python -u autotrain_models1.py \
    --features-path ../../../data/features/unified-20k/allfeatures \
    --time-limit 3600 \
    --presets medium_quality_faster_train \
    2>&1 | tee autogluon_allfeatures-20k.out

# AutoGluon (selected features)
python -u autotrain_models1.py \
    --features-path ../../../data/features/unified-20k/selected \
    --time-limit 3600 \
    --presets medium_quality_faster_train \
    2>&1 | tee autogluon_selected-20k.out
```

### Model 2 (5-Class Classification)
```bash

conda activate ts-sktime
# RAW mode
python -u train_model2.py \
    --mode raw \
    --data-path ../../../data/raw/unified-20k \
    2>&1 | tee train2_raw-20k.out

# FEATURES mode (all features)
# Note: Model 2 automatically filters out stationary samples
python -u train_model2.py \
    --mode features \
    --features-path ../../../data/features/unified-20k/allfeatures \
    2>&1 | tee train2_allfeatures-20k.out

# FEATURES mode (selected features)
python -u train_model2.py \
    --mode features \
    --features-path ../../../data/features/unified-20k/selected \
    2>&1 | tee train2_selected-20k.out

conda activate ts-autogluon
# AutoGluon (all features)
python -u autotrain_models2.py \
    --features-path ../../../data/features/unified-20k/allfeatures \
    --time-limit 3600 \
    --presets medium_quality_faster_train \
    2>&1 | tee autogluon2_allfeatures-20k.out

# AutoGluon (selected features)
python -u autotrain_models2.py \
    --features-path ../../../data/features/unified-20k/selected \
    --time-limit 3600 \
    --presets medium_quality_faster_train \
    2>&1 | tee autogluon2_selected-20k.out
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

##  License

This project is licensed under the MIT License.

---

## 👤 Author

**Ismail Guzel**
- GitHub: [@ismailguzel](https://github.com/ismailguzel)

---

##  References

- TSFresh: https://tsfresh.readthedocs.io/
- scikit-learn: https://scikit-learn.org/
- sktime: https://www.sktime.net/

---

## 📞 Support

For questions or issues, please open an issue on GitHub.

---

## 🎉 Acknowledgments

- TSFresh team for excellent feature engineering library
- scikit-learn community for machine learning tools
- sktime for time series classification algorithms