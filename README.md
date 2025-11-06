# Hierarchical Time Series Classification

A pipeline for hierarchical classification of time series data using TSFresh feature engineering and machine learning models.

## Project Overview

This project implements a hierarchical classification system for time series data:

1. **Level 1 (Binary)**: Stationary vs Non-Stationary
2. **Level 2 (5-Class)**: Trend, Volatility, Stochastic, Anomaly, Structural Break

### Key Features
- Automated time series generation (test: 1.5K, full: 90K samples)
- Three training approaches:
  - **RAW mode**: Direct time series classification with sktime (MiniROCKET, Arsenal, etc.)
  - **FEATURES mode**: TSFresh features + sklearn classifiers (XGBoost, SVM, etc.)
  - **AutoTrain mode**: AutoML with AutoGluon or PyCaret
- Hierarchical two-stage classification pipeline
- TRUBA/HPC support with SLURM scripts
- Comprehensive evaluation and testing tools

---

## 📂 Project Structure

```
hierarchical-ts-classification/
├── 01-data-generation/          # Data generation
│   ├── generate.py              # Main dataset generation (90K)
│   ├── generate_toy.py          # Test dataset (1.5K)
│   ├── verify_ids.py            # Data integrity verification
│   ├── config.py                # Dataset configuration (full + test)
│   └── jobs-slurm/              # TRUBA SLURM scripts
│
├── 02-preprocessing/            # Feature engineering (OPTIONAL)
│   ├── extract_features.py     # TSFresh feature extraction
│   ├── feature_selection.py    # Feature selection methods
│   └── README.md                # Preprocessing docs
│
├── 03-models/hierarchical/      # Hierarchical models
│   ├── model1_binary/           # Level 1: Binary (Stat vs Non-stat)
│   │   ├── train_model1.py      # Dual-mode training (RAW/FEATURES)
│   │   ├── autotrain_models1.py # AutoML training (AutoGluon/PyCaret)
│   │   ├── test_model1.py       # Testing script
│   │   ├── slurm_train_model1.sh # TRUBA SLURM script
│   │   └── README.md            # Documentation
│   │
│   ├── model2_nonstationary/    # Level 2: 5-Class (Non-stat types)
│   │   ├── train_model2.py      # Dual-mode training (RAW/FEATURES)
│   │   ├── autotrain_models2.py # AutoML training (AutoGluon/PyCaret)
│   │   ├── test_model2.py       # Testing script
│   │   ├── slurm_train_model2.sh # TRUBA SLURM script
│   │   └── README.md            # Documentation
│   │
│   ├── submit_training.sh       # Interactive SLURM job helper
│   └── TRUBA_TRAINING_GUIDE.md  # Complete HPC guide
│
├── data/                        # Data storage
│   ├── raw/                     # Raw time series
│   ├── features/                # TSFresh features (optional)
│   └── README.md                # Data documentation
│
├── README.md                    # This file
└── requirements.txt             # Dependencies
```

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/ismailguzel/hierarchical-ts-classification.git
cd hierarchical-ts-classification

# Install dependencies
pip install -r requirements.txt
```

### 2. Smoke Test (Verify Setup)

```bash
# Verify installation and setup
bash smoke_test.sh

# With options
bash smoke_test.sh --mode features --with-model2
```

### 3. Generate Data

```bash
# Generate test dataset (1,450 samples, recommended for development)
cd 01-data-generation
python generate_toy.py

# Verify data integrity
python verify_ids.py --data-path ../data/raw/unified-test

# OR generate full dataset (90,000 samples, for production)
python generate.py
```

### 4. Train Models (RAW Mode - Recommended)

RAW mode works directly on time series without feature extraction:

```bash
# Train Model 1 (Binary: Stationary vs Non-Stationary)
cd ../03-models/hierarchical/model1_binary
python train_model1.py --mode raw --classifier minirocket

# Test Model 1
python test_model1.py \
    --model-path saved_models/model1_binary_minirocket_*.pkl \
    --data-path ../../../data/raw/unified-test

# Train Model 2 (5-Class: Non-Stationary Types)
cd ../model2_nonstationary
python train_model2.py --mode raw --classifier minirocket

# Test Model 2
python test_model2.py \
    --model-path saved_models/model2_nonstationary_minirocket_*.pkl \
    --data-path ../../../data/raw/unified-test
```

### 5. Alternative: FEATURES Mode (Optional)

Use TSFresh features with traditional ML classifiers:

```bash
# Step 5a: Extract Features
cd 02-preprocessing
python extract_features.py \
    --input ../data/raw/unified-test \
    --output ../data/features \
    --feature-set efficient \
    --n-jobs 4

# Step 5b: Select Features
python feature_selection.py \
    --input ../data/features \
    --output ../data/features/selected \
    --method mutual_info \
    --n-features 100 \
    --target all

# Step 5c: Train with Features
cd ../03-models/hierarchical/model1_binary
python train_model1.py --mode features \
    --classifier xgboost \
    --features-path ../../../data/features/selected
```

### 6. Alternative: AutoTrain Mode (AutoML)

Automated model selection and hyperparameter optimization:

```bash
# Prerequisites: Extract and select features first (see step 5a-5b)

# AutoGluon (recommended for best performance)
cd 03-models/hierarchical/model1_binary
python autotrain_models1.py --engine autogluon \
    --features-path ../../../data/features/selected \
    --time-limit 3600

# PyCaret (for model comparison)
python autotrain_models1.py --engine pycaret \
    --features-path ../../../data/features/selected \
    --folds 5
```

### 7. TRUBA/HPC Usage

For large-scale training on TRUBA cluster:

```bash
# Interactive job submission
cd 03-models/hierarchical
bash submit_training.sh

# Or submit directly
cd model1_binary
sbatch slurm_train_model1.sh
```

See `03-models/hierarchical/TRUBA_TRAINING_GUIDE.md` for detailed HPC setup.

### 8. Smoke Test (Full Validation)

```bash
# Basic (raw mode). Uses full dataset if available; falls back to test dataset.
bash smoke_test.sh

# Features mode
bash smoke_test.sh --mode features

# Include Model 2 quick check
bash smoke_test.sh --with-model2

# Customize
bash smoke_test.sh --data-path data/raw/unified-test --n-samples 50 --classifier tsf
```

---

## Pipeline Overview

### Stage 1: Data Generation
- Generate synthetic time series data using `ts-stationary` library
- Two datasets: Test (1,450 samples) and Full (90,000 samples)
- Categories: stationary, deterministic_trends, volatility, stochastic, anomalies, structural_breaks
- Verification with `verify_ids.py` for data integrity

### Stage 2: Feature Engineering (OPTIONAL - for FEATURES/AutoTrain modes)
- Extract time series features using **TSFresh**
- Feature set size configurable (minimal/efficient/comprehensive)
- Statistical, temporal, frequency, and complexity features
- **Not needed for RAW mode** - models work directly on time series

### Stage 3: Feature Selection (OPTIONAL - for FEATURES/AutoTrain modes)
- Multiple selection methods: mutual information, statistical tests, importance
- Reduce dimensionality while maintaining performance
- Task-specific feature selection (binary, primary, sub)
- Creates standardized output structure per task

### Stage 4: Model Training (Three Approaches)

#### A. RAW Mode (Recommended)
- Direct time series classification with sktime
- Classifiers: TimeSeriesForest, ROCKET, MiniROCKET, Arsenal, ShapeletTransform
- No feature engineering required
- Fast and effective

#### B. FEATURES Mode
- Traditional ML with TSFresh features
- Classifiers: Random Forest, XGBoost, SVM
- Feature importance analysis
- Interpretable features

#### C. AutoTrain Mode (AutoML)
- Automated model selection and hyperparameter optimization
- Engines: AutoGluon (ensemble/stacking) or PyCaret (model comparison)
- Requires pre-extracted features
- Best for production-ready models

### Stage 5: Hierarchical Classification
- **Model 1 (Binary)**: Classify as Stationary (0) or Non-Stationary (1)
- **Model 2 (5-Class)**: If Non-Stationary, classify into:
  - 0: Trend
  - 1: Volatility
  - 2: Stochastic
  - 3: Anomaly
  - 4: Structural Break

### Stage 6: Evaluation
- Test scripts for each model (`test_model1.py`, `test_model2.py`)
- Comprehensive metrics: accuracy, precision, recall, F1-score
- Confusion matrices and classification reports
- Per-class performance analysis

---

## Configuration

### Data Generation

Edit `01-data-generation/config.py` to customize:

- Dataset size per category (COUNTS_90K for full, COUNTS_TEST for test)
- Time series length (min/max)
- Generation parameters
- Random seed for reproducibility

Verify data with:
```bash
python verify_ids.py --data-path ../data/raw/unified-test
```

### Feature Extraction (Optional - for FEATURES/AutoTrain modes)

Choose feature set in `02-preprocessing/extract_features.py`:

- `minimal`: ~20 features, fast
- `efficient`: ~200 features, balanced (Recommended)
- `comprehensive`: ~800 features, slow

### Model Training

#### RAW Mode
All training scripts support command-line arguments:

```bash
# RAW mode (default, recommended)
python train_model1.py --mode raw --classifier minirocket

# Available classifiers: tsf, rocket, minirocket, arsenal, shapelet, hivecote
```

#### FEATURES Mode
```bash
# FEATURES mode (traditional ML)
python train_model1.py --mode features \
    --classifier xgboost \
    --features-path /path/to/features

# Available classifiers: rf, xgboost, svm
```

#### AutoTrain Mode
```bash
# AutoGluon (ensemble + stacking)
python autotrain_models1.py --engine autogluon \
    --features-path /path/to/features \
    --time-limit 3600

# PyCaret (model comparison)
python autotrain_models1.py --engine pycaret \
    --features-path /path/to/features \
    --folds 5
```

### TRUBA/HPC Configuration

For cluster training, see:
- `03-models/hierarchical/TRUBA_TRAINING_GUIDE.md`: Complete HPC setup guide
- `submit_training.sh`: Interactive job submission helper
- `slurm_train_model*.sh`: Individual SLURM scripts

---

<!-- Expected results and runtime estimates intentionally omitted to keep README usage-focused. -->

---

## Dependencies

Core dependencies:

```
# Time series and ML
tsfresh>=0.20.0
scikit-learn>=1.3.0
sktime>=0.24.0
pandas>=2.0.0
numpy>=1.24.0

# Optional (AutoTrain mode)
autogluon.tabular>=0.8.0
pycaret>=3.0.0
```

For complete list, see `requirements.txt`.

Install all:
```bash
pip install -r requirements.txt
```

Install with AutoML support:
```bash
pip install -r requirements.txt
pip install autogluon.tabular pycaret
```

---

## Documentation

### Quick Reference
- **Main README**: Project overview and quick start (this file)
- **Data Generation**: `01-data-generation/README.md`
- **Preprocessing**: `02-preprocessing/README.md`
- **Model Training**: `03-models/hierarchical/README.md`
- **Model 1 (Binary)**: `03-models/hierarchical/model1_binary/README.md`
- **Model 2 (5-Class)**: `03-models/hierarchical/model2_nonstationary/README.md`
- **TRUBA/HPC Guide**: `03-models/hierarchical/TRUBA_TRAINING_GUIDE.md`

### Training Modes Comparison

| Mode | Input | Classifiers | Preprocessing | Use Case |
|------|-------|-------------|---------------|----------|
| RAW | Time series | sktime (ROCKET, Arsenal, etc.) | None | Quick start, best baseline |
| FEATURES | TSFresh features | sklearn (XGBoost, SVM, etc.) | Feature extraction | Interpretability, feature analysis |
| AutoTrain | TSFresh features | AutoML (AutoGluon, PyCaret) | Feature extraction | Production, automated optimization |

### Example Workflows

#### Quick Development Workflow (RAW mode)
```bash
# 1. Generate test data
cd 01-data-generation && python generate_toy.py

# 2. Train Model 1
cd ../03-models/hierarchical/model1_binary
python train_model1.py --mode raw --classifier minirocket

# 3. Test Model 1
python test_model1.py --model-path saved_models/model1_binary_minirocket_*.pkl
```

#### Production Workflow (AutoTrain mode)
```bash
# 1. Generate full dataset
cd 01-data-generation && python generate.py

# 2. Extract and select features
cd ../02-preprocessing
python extract_features.py --input ../data/raw/unified-90k --output ../data/features
python feature_selection.py --input ../data/features --output ../data/features/selected

# 3. Train with AutoML
cd ../03-models/hierarchical/model1_binary
python autotrain_models1.py --engine autogluon --features-path ../../../data/features/selected
```

#### TRUBA/HPC Workflow
```bash
# 1. Generate data on HPC
cd 01-data-generation/jobs-slurm && sbatch full-data-job.sh

# 2. Submit training jobs
cd ../../03-models/hierarchical && bash submit_training.sh
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 License

This project is licensed under the MIT License.

---

## 👤 Author

**Ismail Guzel**
- GitHub: [@ismailguzel](https://github.com/ismailguzel)

---

## 🔗 References

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
