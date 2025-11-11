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
- Three training modes: RAW (sktime), FEATURES (sklearn), AutoTrain (AutoML)
- Hierarchical two-stage classification

---

## 📂 Project Structure

```
hierarchical-ts-classification/
├── 01-data-generation/          # Data generation
│   ├── generate.py              # Scalable dataset generation (5K-200K)
│   ├── verify_ids.py            # Data integrity verification
│   ├── config.py                # Dataset configuration (11 scales)
│   └── jobs-slurm/              # Optional SLURM job scripts
│
├── 02-preprocessing/            # Feature engineering (OPTIONAL)
│   ├── extract_features.py     # TSFresh feature extraction (sequential)
│   ├── extract_dask.py          # TSFresh feature extraction (Dask parallel)
│   ├── feature_selection.py    # Feature selection methods
│   └── README.md                # Preprocessing docs
│
├── 03-models/hierarchical/      # Hierarchical models
│   ├── model1_binary/           # Level 1: Binary (Stat vs Non-stat)
│   │   ├── train_model1.py      # Dual-mode training (RAW/FEATURES)
│   │   ├── autotrain_models1.py # AutoML training (AutoGluon/PyCaret)
│   │   ├── test_model1.py       # Testing script
│   │   ├── slurm_train_model1.sh # Example SLURM script
│   │   └── README.md            # Documentation
│   │
│   ├── model2_nonstationary/    # Level 2: 5-Class (Non-stat types)
│   │   ├── train_model2.py      # Dual-mode training (RAW/FEATURES)
│   │   ├── autotrain_models2.py # AutoML training (AutoGluon/PyCaret)
│   │   ├── test_model2.py       # Testing script
│   │   ├── slurm_train_model2.sh # Example SLURM script
│   │   └── README.md            # Documentation
│   │
│   └── submit_training.sh       # Interactive SLURM job helper
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
python test_model1.py --model-path saved_models/model1_binary_*.pkl
python test_model2.py --model-path saved_models/model2_nonstationary_*.pkl
```

### Alternative Workflows

**FEATURES Mode** (TSFresh + sklearn):
```bash
cd 02-preprocessing
python extract_features.py --input ../data/raw/unified-test
python feature_selection.py --method mutual_info
cd ../03-models/hierarchical/model1_binary
python train_model1.py --mode features
```

**AutoTrain Mode** (AutoML):
```bash
# After feature extraction and selection
cd 03-models/hierarchical/model1_binary

# Using AutoGluon (recommended)
python autotrain_models1.py \
    --engine autogluon \
    --features-path ../../../data/features/selected \
    --time-limit 3600 \
    --presets medium_quality_faster_train

# Using PyCaret (alternative)
python autotrain_models1.py \
    --engine pycaret \
    --features-path ../../../data/features/selected \
    --folds 5

# Model 2 AutoTrain
cd ../model2_nonstationary
python autotrain_models2.py --engine autogluon --features-path ../../../data/features/selected
```

**Note:** AutoTrain requires feature extraction first (see 02-preprocessing/).

See individual README files for detailed options and parameters.

**Smoke Test**:
```bash
bash smoke_test.sh
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
│    • extract_features.py → TSFresh features (sequential)       │
│    • extract_dask.py → TSFresh features (Dask parallel)        │
│    • feature_selection.py → Select relevant features           │
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
│       → TSFresh features → AutoML (AutoGluon/PyCaret)          │
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
│    • Metrics: Accuracy, Precision, Recall, F1                  │
│    • Confusion matrices and classification reports             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Training Modes

| Mode | Input | Classifiers | When to Use |
|------|-------|-------------|-------------|
| **RAW** | Time series | sktime (ROCKET, Arsenal) | Quick start, best baseline |
| **FEATURES** | TSFresh features | sklearn (XGBoost, SVM) | Feature analysis, interpretability |
| **AutoTrain** | TSFresh features | AutoML (AutoGluon, PyCaret) | Production, automated tuning |

See individual README files for detailed configuration options.

---

## Requirements

```bash
pip install -r requirements.txt

# Optional: AutoML support
pip install autogluon.tabular pycaret
```

Core: `tsfresh`, `scikit-learn`, `sktime`, `pandas`, `numpy`

---

## Documentation

Detailed documentation in subdirectories:

- **Data Generation**: [`01-data-generation/README.md`](01-data-generation/README.md)
- **Preprocessing**: [`02-preprocessing/README.md`](02-preprocessing/README.md)
- **Model Training**: [`03-models/hierarchical/README.md`](03-models/hierarchical/README.md)
- **Model 1 Details**: [`03-models/hierarchical/model1_binary/README.md`](03-models/hierarchical/model1_binary/README.md)
- **Model 2 Details**: [`03-models/hierarchical/model2_nonstationary/README.md`](03-models/hierarchical/model2_nonstationary/README.md)

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
