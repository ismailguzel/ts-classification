# Hierarchical Time Series Classification

A pipeline for hierarchical classification of time series data using TSFresh feature engineering and machine learning models.

## 🎯 Project Overview

This project implements a hierarchical classification system for time series data:

1. **Level 1 (Binary)**: Stationary vs Non-Stationary
2. **Level 2 (5-Class)**: Trend, Volatility, Stochastic, Anomaly, Structural Break

### Key Features
- Automated time series generation
- Two training modes: RAW (sktime) or FEATURES (TSFresh + sklearn)
- Hierarchical classification (2 levels implemented)
- Evaluation with standard metrics

---

## 📂 Project Structure

```
hierarchical-ts-classification/
├── 01-data-generation/          # Data generation
│   ├── generate.py              # Main dataset generation (90K)
│   ├── generate_toy.py          # Test dataset (~1.5K)
│   └── config.py                # Dataset configuration (full + test)
│
├── 02-preprocessing/            # Feature engineering (OPTIONAL)
│   ├── extract_features.py     # TSFresh feature extraction
│   ├── feature_selection.py    # Feature selection methods
│   └── README.md                # Preprocessing docs
│
├── 03-models/hierarchical/      # Hierarchical models
│   ├── model1_binary/           # Level 1: Binary (Stat vs Non-stat)
│   │   ├── train_model1.py      # Dual-mode training
│   │   ├── test_model1.py       # Testing script
│   │   └── README.md            # Documentation
│   │
│   └── model2_nonstationary/    # Level 2: 5-Class (Non-stat types)
│       ├── train_model2.py      # Dual-mode training
│       ├── test_model2.py       # Testing script
│       └── README.md            # Documentation
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

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/ismailguzel/hierarchical-ts-classification.git
cd hierarchical-ts-classification

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data

```bash
# Generate test dataset (small, for quick testing)
cd 01-data-generation
python generate_toy.py

# OR generate full dataset
python generate.py
```

### 3. Extract Features

```bash
# Extract TSFresh features
cd ../02-preprocessing
python extract_features.py \
    --input ../data/raw/unified-test \
    --output ../data/features \
    --feature-set efficient \
    --n-jobs 4
```

### 4. Select Features

```bash
# Select relevant features
python feature_selection.py \
    --input ../data/features \
    --output ../data/features/selected \
    --method mutual_info \
    --n-features 100 \
    --target all
```

### 5. Train Models

```bash
# Train binary classification model
cd ../03-models/hierarchical/model1_binary
python train_model1.py
```

### 6. Evaluate

```bash
# Test trained model
python test_model1.py
```

### 7. Smoke Test (optional)

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

## 📊 Pipeline Overview

### Stage 1: Data Generation
- Generate synthetic time series data using `ts-stationary` library
- Two levels: Stationary vs Non-Stationary (binary), then 5-class non-stationary classification
- Categories: deterministic_trends, volatility, stochastic, anomalies, structural_breaks

### Stage 2: Feature Engineering (OPTIONAL)
- Extract time series features using **TSFresh** (for FEATURES mode)
- Feature set size configurable
- Statistical, temporal, frequency, and complexity features
- **Not needed for RAW mode** - models work directly on time series

### Stage 3: Feature Selection (OPTIONAL)
- Multiple selection methods: mutual information, statistical tests, importance
- Reduce dimensionality while maintaining performance
- Task-specific feature selection
- **Only for FEATURES mode**

### Stage 4: Model Training
- **Dual-mode architecture**: RAW (sktime) or FEATURES (sklearn)
- Hierarchical classification: Model 1 (binary) → Model 2 (5-class)
- RAW mode classifiers: TimeSeriesForest, ROCKET, Arsenal, others
- FEATURES mode classifiers: Random Forest, XGBoost, SVM
- Cross-validation and performance evaluation

### Stage 5: Evaluation
- Comprehensive metrics: accuracy, precision, recall, F1-score
- Confusion matrices and classification reports
- Per-class performance analysis

---

## 🔧 Configuration

### Data Generation

Edit `01-data-generation/config.py` or `config_test.py` to customize:

- Dataset size per category
- Time series length (min/max)
- Generation parameters

### Feature Extraction (Optional - for FEATURES mode)

Choose feature set in `02-preprocessing/extract_features.py`:

- `minimal`: ~20 features, fast
- `efficient`: ~200 features, balanced ⭐ **Recommended**
- `comprehensive`: ~800 features, slow

### Model Training

All training scripts support command-line arguments:

```bash
# RAW mode (default)
python train_model1.py --mode raw --classifier rocket

# FEATURES mode
python train_model1.py --mode features --classifier xgboost --features-path /path/to/features
```

---

<!-- Expected results and runtime estimates intentionally omitted to keep README usage-focused. -->

---

## Dependencies

Core dependencies:

```
tsfresh>=0.20.0
scikit-learn>=1.3.0
sktime>=0.24.0
pandas>=2.0.0
numpy>=1.24.0
```

For complete list, see `requirements.txt`.

---

## Documentation

- **01-data-generation/**: See individual script docstrings
- **02-preprocessing/**: See [02-preprocessing/README.md](02-preprocessing/README.md)
- **03-models/**: See model-specific READMEs under `03-models/hierarchical/`

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
