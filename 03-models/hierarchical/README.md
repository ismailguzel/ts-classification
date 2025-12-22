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

### Automated Training (via run-training.sh)

```bash
# From project root - trains with statistical features by default
bash run-training.sh

# Or specify feature type and scale
bash run.sh training 20k statistical   # Statistical features
bash run.sh training 20k topological   # Topological features
bash run.sh training 20k combined      # Combined features
bash run.sh training 5k statistical    # 5k dataset for testing
```

### Manual Training

#### Statistical Features (Default)
```bash
cd model1_binary
python train_model1.py \
    --mode features \
    --features-path ../../../data/features/unified-20k/statistical_selected

cd ../model2_nonstationary
python train_model2.py \
    --mode features \
    --features-path ../../../data/features/unified-20k/statistical_selected
```

#### Topological Features
```bash
cd model1_binary
python train_model1.py \
    --mode features \
    --features-path ../../../data/features/unified-20k/topological_selected

cd ../model2_nonstationary
python train_model2.py \
    --mode features \
    --features-path ../../../data/features/unified-20k/topological_selected
```

#### Combined Features
```bash
cd model1_binary
python train_model1.py \
    --mode features \
    --features-path ../../../data/features/unified-20k/combined_selected

cd ../model2_nonstationary
python train_model2.py \
    --mode features \
    --features-path ../../../data/features/unified-20k/combined_selected
```

## Model Specifications

### Model 1: Binary Classification

| Metric | Value |
|--------|-------|
| **Task** | Stationary vs Non-Stationary |
| **Classes** | 2 (Binary) |
| **Best Accuracy** | 96.89% (Statistical + XGBoost) |

**Available Classifiers**:
- **FEATURES Mode (Recommended)**: XGBoost, CatBoost, LightGBM, Random Forest, SVM
- **RAW Mode**: TimeSeriesForest, ROCKET, Arsenal

### Model 2: 5-Class Classification

| Metric | Value |
|--------|-------|
| **Task** | Non-Stationary Type Classification |
| **Classes** | 5 (Trend/Volatility/Stochastic/Anomaly/Structural) |
| **Best Accuracy** | 97.81% (Statistical + XGBoost) |

**5 Classes**:
0. **Trend**: Deterministic trend patterns (linear, quadratic, exponential, etc.)
1. **Volatility**: Changing variance (ARCH, GARCH, EGARCH, APARCH)
2. **Stochastic**: Random walk behavior (RW, ARIMA, ARI, IMA)
3. **Anomaly**: Point and collective anomalies
4. **Structural Break**: Sudden regime changes (mean/variance/trend shifts)

**Available Classifiers**:
- **FEATURES Mode (Recommended)**: XGBoost, CatBoost, LightGBM, Random Forest, SVM
- **RAW Mode**: TimeSeriesForest, ROCKET, Arsenal

## Training Modes

Both models support **dual-mode** training:

### 1. FEATURES Mode (Recommended)
Uses extracted features with sklearn/boosting classifiers.
- **Pros**: Fast inference, interpretable features, high accuracy (XGBoost ~97%)
- **Cons**: Requires preprocessing step
- **Workflow**:
  1. Extract features (`bash run-preprocessing.sh`)
  2. Train model (`--mode features`)

**Supported Feature Types**:
- **Statistical** (TSFresh): ~800 raw → 100 selected
- **Topological** (Persistent Homology): 200 raw → 50 selected
- **Combined** (Statistical + Topological): ~1000 raw → 150 selected

### 2. RAW Mode
Uses raw time series with sktime classifiers.
- **Pros**: No feature engineering required
- **Cons**: Slower training/inference, lower accuracy
- **Workflow**:
  1. Generate data (`bash run-generation.sh`)
  2. Train model (`--mode raw`)

## Feature Type Comparison

| Feature Type | Features | Model 1 | Model 2 | Training Time | Notes |
|:-------------|:--------:|:-------:|:-------:|:-------------:|:------|
| **Statistical** | 100 | 96.89% | 97.81% | ~15 min | Baseline, proven |
| **Topological** | 50 | TBD | TBD | ~10 min | Experimental, shape analysis |
| **Combined** | 150 | TBD | TBD | ~20 min | Potential boost from both |

*TBD: To Be Determined - under evaluation*

## Workflow

### Complete Pipeline
```bash
# 1. Generate data
bash run-generation.sh

# 2. Extract features (all 3 types)
bash run-preprocessing.sh

# 3. Train models (choose feature type)
bash run.sh training 20k statistical
bash run.sh training 20k topological
bash run.sh training 20k combined

# 4. Analyze results
bash run.sh postprocessing 20k statistical
```

### Fast Testing (5k Dataset)
```bash
# Complete 5k pipeline with all feature types
bash run-test-5k.sh
```

### Compare Feature Types
```bash
# Train on all 3 feature types
for FEATURE_TYPE in statistical topological combined; do
    bash run.sh training 20k $FEATURE_TYPE
done

# Compare results
cat 03-models/hierarchical/model1_binary/output/model1_binary_features/results_summary.txt
cat 03-models/hierarchical/model1_binary/output/model1_topological/results_summary.txt
cat 03-models/hierarchical/model1_binary/output/model1_combined/results_summary.txt
```

## Usage Recommendations

### For Testing/Prototyping
- Use **5k dataset**: `bash run-test-5k.sh`
- Use **statistical features**: Proven baseline
- Training time: ~30-60 minutes

### For Production
- Use **20k dataset**: `bash run.sh all 20k`
- Use **XGBoost** with **statistical or combined features**
- Expected accuracy: 96-98%

### For Research
- Compare all 3 feature types
- Analyze feature importance
- Evaluate topological contribution
Important Notes

1. **Preprocessing Required**: Run `bash run-preprocessing.sh` before training in FEATURES mode
2. **Feature Types**: Three types available (statistical, topological, combined)
3. **Conda Environments**: 
   - `ts-sktime` for training (automatically activated)
   - `ts-top` for topological extraction (automatically switched during preprocessing)
4. **Memory Usage**: Model 2 (5-class) requires more RAM than Model 1
5. **Parallel Training**: Model 1 and Model 2 can be trained independently
6. **Output Naming**: Automatically named based on feature type and dataset size

## Troubleshooting

### "Dataset not found" or "Features not found"
```bash
# Generate dataset and extract features
bash run-generation.sh
bash run-preprocessing.sh
```

### "No such file: statistical_selected/"
```bash
# Run full preprocessing (creates all 3 feature types)
bash run-preprocessing.sh
```

### Out of memory error
- Use 5k dataset for testing: `bash run-test-5k.sh`
- Reduce number of classifiers: edit training scripts
- Increase system RAM or use chunking

### Different accuracy than expected
- **Statistical**: Should match ~97% (baseline)
- **Topological**: Expected 85-90% (experimental)
- **Combined**: Target >97% (potential boost)
- Check preprocessing completed correctly
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
