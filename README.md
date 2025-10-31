# Hierarchical Time Series Classification

A comprehensive pipeline for hierarchical classification of time series data using TSFresh feature engineering and machine learning models.

## 🎯 Project Overview

This project implements a hierarchical classification system for time series data:

1. **Level 1 (Binary)**: Stationary vs Non-Stationary
2. **Level 2 (Primary Category)**: AR, MA, ARMA, Random Walk, Trend, etc.
3. **Level 3 (Sub-Category)**: Specific types within each category

### Key Features
- ✅ Automated time series generation (150K+ samples)
- ✅ TSFresh feature engineering (~200-800 features)
- ✅ Feature selection for optimal performance
- ✅ Hierarchical classification models
- ✅ Comprehensive evaluation metrics

---

## 📂 Project Structure

```
hierarchical-ts-classification/
├── 01-data-generation/          # Data generation scripts
│   ├── generate.py              # Generate 150K dataset
│   ├── generate_test.py         # Generate test dataset
│   ├── config_150k.py           # Configuration for 150K dataset
│   └── config_test.py           # Configuration for test dataset
│
├── 02-preprocessing/            # Feature engineering
│   ├── extract_features.py     # TSFresh feature extraction
│   ├── feature_selection.py    # Feature selection methods
│   └── README.md                # Preprocessing documentation
│
├── 03-models/                   # Model training & evaluation
│   └── hierarchical/            # Hierarchical models
│       ├── model1_binary/       # Binary classification (stationary vs non-stationary)
│       ├── configs/             # Model configurations
│       └── MODEL1_QUICKSTART.md # Quick start guide
│
├── data/                        # Data storage
│   ├── raw/                     # Raw generated data
│   ├── processed/               # Processed data
│   └── features/                # Extracted features
│       └── selected/            # Selected features
│
├── README.md                    # This file
└── requirements.txt             # Python dependencies
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
python generate_test.py

# OR generate full 150K dataset (takes 12-18 hours)
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

---

## 📊 Pipeline Overview

### Stage 1: Data Generation
- Generate synthetic time series data using `ts-stationary` library
- Support for multiple categories: AR, MA, ARMA, Random Walk, Trends, etc.
- Balanced dataset with configurable sample sizes

### Stage 2: Feature Engineering
- Extract time series features using **TSFresh**
- 200-800 features per series (configurable)
- Statistical, temporal, frequency, and complexity features

### Stage 3: Feature Selection
- Multiple selection methods: mutual information, statistical tests, importance
- Reduce dimensionality while maintaining performance
- Task-specific feature selection

### Stage 4: Model Training
- Hierarchical classification architecture
- Multiple algorithms: Random Forest, XGBoost, SVM, etc.
- Cross-validation and hyperparameter tuning

### Stage 5: Evaluation
- Comprehensive metrics: accuracy, precision, recall, F1-score
- Confusion matrices and classification reports
- Model comparison and analysis

---

## 🔧 Configuration

### Data Generation

Edit `01-data-generation/config_150k.py` or `config_test.py`:

```python
# Dataset size
COUNTS_150K = {
    'stationary': {
        'ar': 10000,
        'ma': 10000,
        # ...
    },
    'unstationary': {
        'random_walk': 10000,
        # ...
    }
}

# Time series length
LENGTH_CONFIG = {
    'min': 1000,
    'max': 5000
}
```

### Feature Extraction

Choose feature set in `02-preprocessing/extract_features.py`:

- `minimal`: ~20 features, fast
- `efficient`: ~200 features, balanced ⭐ **Recommended**
- `comprehensive`: ~800 features, slow

### Model Training

Configure models in `03-models/hierarchical/configs/`:

- Learning rate, epochs, batch size
- Model architecture
- Cross-validation settings

---

## 📈 Expected Results

### Binary Classification (Stationary vs Non-Stationary)
- **Accuracy**: >95%
- **Training Time**: 5-15 minutes
- **Dataset**: 15,000 samples

### Primary Category Classification
- **Accuracy**: >85%
- **Training Time**: 10-30 minutes
- **Dataset**: 15,000 samples

### Sub-Category Classification
- **Accuracy**: >75%
- **Training Time**: 15-45 minutes
- **Dataset**: 150,000 samples

---

## 🛠️ Dependencies

Core dependencies:

```
tsfresh>=0.20.0          # Feature engineering
scikit-learn>=1.3.0      # Machine learning
sktime>=0.24.0           # Time series algorithms
pandas>=2.0.0            # Data manipulation
numpy>=1.24.0            # Numerical computing
```

For complete list, see `requirements.txt`.

---

## 📚 Documentation

- **01-data-generation/**: See individual script docstrings
- **02-preprocessing/**: See [02-preprocessing/README.md](02-preprocessing/README.md)
- **03-models/**: See [03-models/hierarchical/MODEL1_QUICKSTART.md](03-models/hierarchical/MODEL1_QUICKSTART.md)

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
