# Hierarchical Time Series Classification Models

This directory contains training scripts for the hierarchical time series classification system.

## Directory Structure

```
03-models/hierarchical/
├── model1_binary/              # Binary classification (Stationary vs Non-Stationary)
│   ├── train_model1.py        # Training script
│   └── README.md              # Model 1 documentation
│
├── model2_nonstationary/       # 5-class classification (Non-Stationary types)
│   ├── train_model2.py        # Training script
│   └── README.md              # Model 2 documentation
│
└── utils/                      # Shared utilities
    ├── data_utils.py          # Data loading functions
    ├── metrics.py             # Evaluation metrics
    └── constants.py           # Shared constants
```

## Hierarchical System

```
                    ┌─────────────────┐
                    │  Input Series   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │    MODEL 1      │
                    │  (Binary Class) │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
      ┌───────▼────────┐          ┌────────▼────────┐
      │  STATIONARY    │          │ NON-STATIONARY  │
      │   (Class 0)    │          │   (Class 1)     │
      └────────────────┘          └────────┬────────┘
                                           │
                                  ┌────────▼────────┐
                                  │    MODEL 2      │
                                  │  (5-Class)      │
                                  └────────┬────────┘
                                           │
                        ┌──────────────────┼──────────────────┐
                        │         │        │        │         │
                   ┌────▼───┐ ┌──▼──┐ ┌──▼───┐ ┌──▼───┐ ┌──▼─────┐
                   │ Trend  │ │Volat│ │Stoch │ │Anomly│ │Struct  │
                   │(Class0)│ │(Cl1)│ │(Cl2) │ │(Cl3) │ │(Class4)│
                   └────────┘ └─────┘ └──────┘ └──────┘ └────────┘
```

## Quick Start

### Local Environment (Test/Development)

```bash
# Train Model 1 (Binary)
cd model1_binary
python train_model1.py --mode features --classifier xgboost

# Train Model 2 (5-Class)
cd model2_nonstationary
python train_model2.py --mode features --classifier xgboost
```

## Model Specifications

### Model 1: Binary Classification

| Metric | Value |
|--------|-------|
| **Task** | Stationary vs Non-Stationary |
| **Classes** | 2 (Binary) |

**Available Classifiers**:
- **FEATURES Mode (Recommended)**: XGBoost, Random Forest, CatBoost, SVM
- **RAW Mode**: TimeSeriesForest, ROCKET, Arsenal

### Model 2: 5-Class Classification

| Metric | Value |
|--------|-------|
| **Task** | Non-Stationary Type Classification |
| **Classes** | 5 (Trend/Volatility/Stochastic/Anomaly/Structural) |

**5 Classes**:
0. **Trend**: Deterministic trend patterns
1. **Volatility**: Changing variance
2. **Stochastic**: Random walk behavior
3. **Anomaly**: Point and collective anomalies
4. **Structural Break**: Sudden regime changes

**Available Classifiers**:
- **FEATURES Mode (Recommended)**: XGBoost, Random Forest, CatBoost, SVM
- **RAW Mode**: TimeSeriesForest, ROCKET, Arsenal
- ROCKET (2000 kernels for 5-class)
- Arsenal (2000 kernels)

## Training Modes

Both models support **dual-mode** training:

### 1. FEATURES Mode (Recommended)
Uses TSFresh extracted features with sklearn/boosting classifiers.
- **Pros**: Fast inference, interpretable features, high accuracy (XGBoost ~97%)
- **Workflow**:
  1. Extract features (`02-preprocessing/feature_extraction.py`)
  2. Select features (`02-preprocessing/feature_selection.py`)
  3. Train model (`--mode features`)

### 2. RAW Mode
Uses raw time series with sktime classifiers.
- **Pros**: No feature engineering required
- **Cons**: Slower training/inference for complex models like HIVECOTE
- **Workflow**:
  1. Generate data
  2. Train model (`--mode raw`)

## Usage Recommendations

### For Testing/Prototyping
```bash
# Use TimeSeriesForest or ROCKET in RAW mode
python train_model1.py --mode raw --classifier tsf
```

### For Production
```bash
# Use XGBoost in FEATURES mode
python train_model1.py --mode features --classifier xgboost
```

### For Benchmarking
```bash
# Train all and compare
python train_model1.py --mode features --classifier all
```

## Model Evaluation

Training scripts automatically evaluate models on the test set and save:
- Predictions (CSV)
- Metrics (JSON)
- Misclassified samples (CSV)
- Feature importance (for FEATURES mode)

Results are saved in `saved_models/model*_<mode>/` directories.

Test scripts calculate:
- Evaluation metrics
- Confusion matrix
- Per-class metrics
- Classification report

## More Information

- **Model 1 Details**: `model1_binary/README.md`
- **Model 2 Details**: `model2_nonstationary/README.md`

## Workflow

```bash
# 1. Data Generation (01-data-generation/)
python generate.py

# 2. Feature Extraction (02-preprocessing/)
python feature_extraction.py

# 3. Model training (03-models/hierarchical/)
python train_model1.py --mode features
python train_model2.py --mode features

# 4. Testing (03-models/hierarchical/)
python test_model1.py --model-path models/model1_xgboost_features.pkl
python test_model2.py --model-path models/model2_xgboost_features.pkl
```

## Important Notes

1. **Dataset Required**: Generate data using `01-data-generation/` before training.
2. **HIVECOTEV2 is slow**: Use this classifier only for research/benchmarking.
3. **Memory Usage**: Model 2 (5-class) requires more RAM.
4. **Parallel Training**: Model 1 and Model 2 can be trained in parallel.

## Troubleshooting

### "Dataset not found" error

```bash
# Generate dataset
cd ../../01-data-generation
python generate.py
```

### Out of memory error

```bash
# Increase test-size (reduces training data)
python train_model1.py --test-size 0.3  # default 0.2

# Or use a smaller dataset scale
python train_model1.py --data-path ../../../data/raw/unified-test
```

### sktime import error

```bash
# Update sktime
pip install --upgrade sktime

# Or use a classifier without issues
python train_model1.py --classifier minirocket
```

---

**Prepared by**: GitHub Copilot  
**Last Update**: December 12, 2025  
**Python**: >= 3.10  
**sktime**: >= 0.24.0  
**scikit-learn**: >= 1.3.0
