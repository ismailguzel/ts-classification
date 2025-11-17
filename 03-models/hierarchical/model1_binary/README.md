# Model 1: Binary Classification

##  Objective

Classify time series as **Stationary (0)** or **Non-Stationary (1)**.

This is the first level of the hierarchical classification system.

---

## Training Modes

### Mode 1: RAW (Default) - sktime classifiers
Uses raw time series with time series classifiers:
- TimeSeriesForest
- ROCKET
- Arsenal
- ShapeletTransform

### Mode 2: FEATURES - sklearn classifiers
Uses TSFresh extracted features with traditional ML:
- Random Forest
- XGBoost
- SVM (Linear)

---

## Models Available

### RAW Mode (sktime)

#### 1. TimeSeriesForest
- Interval-based ensemble
- Fast baseline

#### 2. ROCKET
- Convolutional kernel transform
- Balanced approach

#### 3. Arsenal
- ROCKET ensemble

### FEATURES Mode (sklearn)

#### 1. Random Forest
- Decision tree ensemble

#### 2. XGBoost
- Gradient boosting

#### 3. SVM (Linear)
- Support vector machine

---

## Usage

### Training - RAW Mode (Default)

```bash
cd 03-models/hierarchical/model1_binary

# Train all models (default)
python train_model1.py --mode raw

# Train specific classifier
python train_model1.py --mode raw --classifier rocket
python train_model1.py --mode raw --classifier arsenal
python train_model1.py --mode raw --classifier tsf
```

Available classifiers:
- `all` (default)
- `tsf`
- `rocket`
- `arsenal`

Requirements:
- Raw data: `../../../data/raw/unified-5k/` (default)
- Output: `saved_models/model1_binary_raw_<classifier>/`

### Training - FEATURES Mode

```bash
# First: Extract and select features
cd ../../../02-preprocessing
python extract_features.py --input ../data/raw/unified-5k --output ../data/features/unified-5k/allfeatures
python feature_selection.py --input ../data/features/unified-5k/allfeatures --output ../data/features/unified-5k/selected --target binary

# Then: Train with features
cd ../03-models/hierarchical/model1_binary
python train_model1.py --mode features --features-path ../../../data/features/unified-5k/selected
```

Requirements:
- Features: `../../../data/features/unified-5k/selected/binary/` (default)
- Output: `saved_models/model1_binary_features/`

### Testing

```bash
# Test RAW mode model
python test_model1.py --model-path saved_models/model1_binary_raw_rocket/model1_binary_classifier.pkl

# Test FEATURES mode model
python test_model1.py --model-path saved_models/model1_binary_features/model1_binary_classifier.pkl

# Test with more samples
python test_model1.py --model-path saved_models/model1_binary_raw_rocket/model1_binary_classifier.pkl --n-samples 500
```

Quick test on a subset to verify model works.

---

<!-- Expected performance and recommendations removed to keep README usage-focused. -->

---

## Technical Details

RAW Mode - Data Preparation

1. **Data Source**: Parquet files with metadata
      - Column detection: 'series_id'
   - Data column: 'data' or 'value'
   - Label from: 'is_stationary' metadata
2. **Format**: Univariate time series
3. **Length**: Fixed to 1,500 points (pad/truncate)
4. **Shape**: (n_samples, 1, 1500) for sktime
5. **Split**: 80% train, 20% test (stratified)
6. **Preprocessing**: NaN/inf values replaced with 0

FEATURES Mode - Data Preparation

1. **Data Source**: TSFresh extracted features
2. **Format**: Feature vectors from TSFresh
3. **Features**: 50-150 selected features (mutual info)
4. **Scaling**: StandardScaler normalization
5. **Split**: 80% train, 20% test (stratified)

Model Selection

Script automatically:
1. Trains AutoML on features
2. Automatically selects best model
3. Evaluates on test set
4. Saves results to `output/`

Output Files

### RAW Mode Output Structure
```
saved_models/
└── model1_binary_raw_<classifier>/      # e.g., model1_binary_raw_rocket
    ├── model1_binary_classifier.pkl     # Best trained model
    ├── model1_metadata.pkl              # Training metadata
    ├── model1_raw_<CLASSIFIER>_metrics.json  # Model metrics
    ├── model1_raw_predictions.csv       # Test predictions
    ├── model1_raw_misclassified.csv     # Misclassified samples
    └── model1_raw_summary.json          # Summary comparison
```

### FEATURES Mode Output Structure
```
saved_models/
└── model1_binary_features/              # All sklearn models
    ├── model1_binary_classifier.pkl     # Best trained model
    ├── model1_metadata.pkl              # Training metadata
    ├── scaler.pkl                       # Feature scaler
    ├── predictions_<ModelName>.csv      # Per-model predictions
    ├── misclassified_<ModelName>.csv    # Per-model misclassified
    ├── model1_features_<ModelName>_metrics.json  # Per-model metrics
    ├── model1_features_summary.json     # Summary comparison
    └── catboost_info/                   # CatBoost logs
```

### AutoTrain Mode Output Structure
```
saved_models/
└── model1_binary_autogluon/             # AutoGluon artifacts
    ├── models/                          # Trained model files
    ├── predictions_AutoGluon_Binary.csv
    ├── misclassified_AutoGluon_Binary.csv
    ├── leaderboard_test.csv
    ├── leaderboard_train.csv
    ├── model1_autotrain_metadata.pkl    # Training metadata
    └── metrics/
        └── AutoGluon_Binary_metrics.json
```

Metadata includes:
- Model name and type
- Training mode (raw/features/autotrain)
- Evaluation metrics
- Data shapes and parameters
- Fixed length (for raw mode)
- Scaler (if features mode)
- N_JOBS configuration

---

##  Tips

### When to use RAW mode:
- Quick baseline needed
- Limited computational resources
- Preprocessing pipeline not ready
- Experimenting with time series methods

### When to use FEATURES mode:
- Need interpretable features
- Want to analyze feature importance
- Have computational resources for TSFresh
- Need faster inference time

---

##  Troubleshooting

**Error: Data not found**
```bash
# Generate test data first
cd ../../../01-data-generation
python generate_test.py
```

**Error: Features not found** (features mode)
```bash
# Extract and select features first
cd ../../../02-preprocessing
python extract_features.py
python feature_selection.py --target binary
```

**Training issues**
- Check data quality and label distribution
- Ensure balanced labels (should be ~50/50)
- Try different classifier
- Increase dataset size
- Check for NaN/inf values in data

**Out of memory** (raw mode)
- Reduce fixed_length parameter (currently 1500)
- Use features mode instead
- Data is processed in batches (BATCH_SIZE=10)
- Check available RAM

**ROCKET models fail with NumPy 2.0**
- Error: AttributeError: np.NINF removed
- Solution: Downgrade NumPy: `pip install "numpy<2.0"`
- Or use TimeSeriesForest/ShapeletTransform instead

---

##  Next Steps

After successful Model 1 training:

1. **Review Results**: Check classification report and confusion matrix
2. **Test Model**: Run `python test_model1.py`
3. **Compare Modes**: Try both raw and features modes
4. **Proceed to Model 2**: Train 5-class non-stationary classifier
5. **Build Pipeline**: Combine Model 1 + Model 2 for hierarchical classification

---

##  Hierarchical Pipeline

Model 1 is part of a hierarchical system:

```
Input Time Series
      ↓
[Model 1: Binary] ← YOU ARE HERE
Is Stationary?
      ↓
  YES → Stationary (stop)
      ↓
   NO → Continue to Model 2
      ↓
[Model 2: 5-Class]
What type of non-stationary?
  0: Trend
  1: Volatility
  2: Stochastic
  3: Anomaly
  4: Structural Break
```

---

##  References

- **sktime**: https://www.sktime.net/
- **ROCKET**: Dempster et al., 2020
- **TSFresh**: https://tsfresh.readthedocs.io/
- **ts-stationary**: Data generation library

---

## 🆘 Configuration

### Parallelization Settings

Both training files have centralized `N_JOBS` configuration:

```python
# At the top of train_model1.py
N_JOBS = -1  # Use all available cores
```

To change parallelization, edit this single variable.

### Data Format

The code automatically detects:
- **Column names**: 'series_id' for series identifier
- **Data columns**: 'data' or 'value' for time series values
- **Labels**: 'is_stationary' metadata (True/False)

### Batch Processing

For memory efficiency, data is loaded in batches:
- **BATCH_SIZE**: 10 files at a time
- Garbage collection after each batch
- Prevents memory overflow on large datasets

---

**Last Updated:** 2025-11-01

