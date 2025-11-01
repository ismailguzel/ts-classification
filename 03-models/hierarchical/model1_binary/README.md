# Model 1: Binary Classification

## 🎯 Objective

Classify time series as **Stationary (0)** or **Non-Stationary (1)**.

This is the first level of the hierarchical classification system.

---

## 🔄 Training Modes

### Mode 1: RAW (Default) - sktime classifiers
Uses raw time series with specialized time series classifiers:
- ✅ No feature engineering needed
- ✅ Fast training
- ✅ Good baseline performance
- 🌲 **TimeSeriesForest**: Fast ensemble method
- 🚀 **ROCKET**: State-of-the-art (1000 kernels for binary)
- ⚡ **MiniROCKET**: 10x faster than ROCKET
- 🎯 **Arsenal**: ROCKET-based ensemble (1000 kernels)
- 🔍 **ShapeletTransform**: Pattern-based classification
- 🏆 **HIVECOTEV2**: Most powerful (very slow)

### Mode 2: FEATURES - sklearn classifiers
Uses TSFresh extracted features with traditional ML:
- ✅ Potentially higher accuracy
- ✅ More interpretable features
- ✅ Faster inference
- 🌲 **Random Forest**: Robust ensemble
- 🚀 **XGBoost**: Gradient boosting
- ⚡ **SVM (Linear)**: Linear classifier

---

## 📊 Models Available

### RAW Mode (sktime)

#### 1. TimeSeriesForest
- **Type**: Interval-based ensemble
- **Speed**: Fast ⚡⚡⚡
- **Accuracy**: 93-95%
- **Use case**: Quick baseline

#### 2. ROCKET
- **Type**: Convolutional kernel transform
- **Kernels**: 1000 (optimized for binary)
- **Speed**: Fast ⚡⚡
- **Accuracy**: 95-97%
- **Use case**: Balanced performance

#### 3. MiniROCKET ⭐ Recommended
- **Type**: Faster ROCKET variant
- **Speed**: Very fast ⚡⚡⚡
- **Accuracy**: 95-97%
- **Use case**: Best speed/accuracy ratio

#### 4. Arsenal ⭐ Best Accuracy
- **Type**: ROCKET ensemble (1000 kernels)
- **Speed**: Moderate ⚡
- **Accuracy**: 96-98%
- **Use case**: Highest accuracy

#### 5. ShapeletTransform
- **Type**: Pattern-based
- **Speed**: Slow ⏱️
- **Accuracy**: 93-96%
- **Use case**: Interpretable patterns

#### 6. HIVECOTEV2
- **Type**: Hybrid ensemble
- **Speed**: Very slow 🐌
- **Accuracy**: 97-99%
- **Use case**: Research/benchmarking only

### FEATURES Mode (sklearn)

#### 1. Random Forest
- **Type**: Decision tree ensemble
- **Speed**: Fast ⚡
- **Use case**: Robust baseline

#### 2. XGBoost
- **Type**: Gradient boosting
- **Speed**: Fast ⚡⚡
- **Performance**: Excellent
- **Use case**: Best performance

#### 3. SVM (Linear)
- **Type**: Support vector machine
- **Speed**: Fast ⚡
- **Use case**: High-dimensional features

---

## 🚀 Usage

### Training - RAW Mode (Default)

```bash
cd 03-models/hierarchical/model1_binary

# Train all models (default)
python train_model1.py --mode raw

# Train specific classifier
python train_model1.py --mode raw --classifier minirocket  # Fastest
python train_model1.py --mode raw --classifier arsenal     # Best accuracy
python train_model1.py --mode raw --classifier rocket      # Balanced
python train_model1.py --mode raw --classifier tsf         # Quick baseline
python train_model1.py --mode raw --classifier shapelet    # Pattern-based
python train_model1.py --mode raw --classifier hivecote    # Research (slow)
```

**Available classifiers:**
- `all` - Train all models (default)
- `tsf` - TimeSeriesForest only
- `rocket` - ROCKET only
- `minirocket` - MiniROCKET only ⭐ **Recommended for speed**
- `arsenal` - Arsenal only ⭐ **Recommended for accuracy**
- `shapelet` - ShapeletTransform only
- `hivecote` - HIVECOTEV2 only (very slow)

**Requirements:**
- Raw data: `../../../data/raw/unified-test/`
- Time (for 1,450 samples): 
  - MiniROCKET: ~20 seconds
  - ROCKET: ~60 seconds
  - Arsenal: ~90 seconds
  - All models: ~5-10 minutes
  - HIVECOTEV2: ~30+ minutes
- Output: `saved_models/model1_binary_classifier.pkl`
- Parallelization: N_JOBS=-1 (uses all CPU cores)

### Training - FEATURES Mode

```bash
# First: Extract and select features
cd ../../../02-preprocessing
python extract_features.py --input ../data/raw/unified-test --output ../data/features
python feature_selection.py --input ../data/features --output ../data/features/selected --target binary

# Then: Train with features
cd ../03-models/hierarchical/model1_binary
python train_model1.py --mode features --features-path ../../../data/features/selected
```

**Requirements:**
- Features: `../../../data/features/selected/features_binary_*.parquet`
- Time: ~5-10 minutes (after feature extraction)
- Output: `saved_models/model1_binary_classifier.pkl`
- Parallelization: N_JOBS=-1 (uses all CPU cores)

### Testing

```bash
# Test with 100 samples (default)
python test_model1.py

# Test with more samples
python test_model1.py --n-samples 500
```

Quick test on 100 samples to verify model works.

---

## 📈 Expected Performance

**Target Accuracy:** 80-90% (depending on classifier)

**Dataset:** 1,450 samples (752 stationary + 698 non-stationary)

### RAW Mode Results (~1,450 test samples)

| Classifier | Accuracy | Training Time | Speed | Use Case |
|------------|----------|---------------|-------|----------|
| TimeSeriesForest | 80-85% | ~30s | ⚡⚡⚡ | Quick baseline |
| ROCKET | 82-87% | ~60s | ⚡⚡ | Balanced |
| **MiniROCKET** ⭐ | 82-87% | ~20s | ⚡⚡⚡ | **Best speed/accuracy** |
| **Arsenal** ⭐ | 84-89% | ~90s | ⚡ | **Best accuracy** |
| Shapelet | 80-86% | ~5min | ⏱️ | Interpretable |
| HIVECOTEV2 | 86-91% | ~30+min | 🐌 | Research only |

### FEATURES Mode Results (~1,450 test samples)

| Classifier | Accuracy | Training Time | Use Case |
|------------|----------|---------------|----------|
| Random Forest | 83-88% | ~5s | Robust baseline |
| XGBoost | 85-90% | ~8s | Best performance |
| SVM Linear | 82-87% | ~3s | High-dimensional |

**Note on Performance:**
Binary classification (stationary vs non-stationary) is relatively straightforward. The achieved accuracy depends on:
- Data quality and diversity
- Series length and complexity
- Classifier choice

**Recommendations:**
- 🏃 **Need speed?** → Use MiniROCKET
- 🎯 **Need accuracy?** → Use Arsenal or XGBoost (features)
- ⚖️ **Balanced?** → Use ROCKET or MiniROCKET
- 🔬 **Research?** → Compare all models

---

## 🔧 Technical Details

### RAW Mode - Data Preparation

1. **Data Source**: Parquet files with metadata
   - Column detection: 'series_id' or 'id'
   - Data column: 'data' or 'value'
   - Label from: 'is_stationary' metadata
2. **Format**: Univariate time series
3. **Length**: Fixed to 1,500 points (pad/truncate)
4. **Shape**: (n_samples, 1, 1500) for sktime
5. **Split**: 80% train, 20% test (stratified)
6. **Preprocessing**: NaN/inf values replaced with 0

### FEATURES Mode - Data Preparation

1. **Data Source**: TSFresh extracted features
2. **Format**: Feature vectors from TSFresh
3. **Features**: 50-150 selected features (mutual info)
4. **Scaling**: StandardScaler normalization
5. **Split**: 80% train, 20% test (stratified)

### Model Selection

Script automatically:
1. Trains multiple models (depending on mode)
2. Compares accuracy on test set
3. Saves best performing model
4. Stores metadata (mode, params, scaler, etc.)

### Output Files

```
saved_models/
├── model1_binary_classifier.pkl    # Best trained model
└── model1_metadata.pkl              # Training metadata
```

Metadata includes:
- Model name and type
- Training mode (raw/features)
- Accuracy metrics
- Data shapes and parameters
- Fixed length (for raw mode)
- Scaler (if features mode)
- N_JOBS configuration

---

## 💡 Tips

### When to use RAW mode:
- Quick baseline needed
- Limited computational resources
- Preprocessing pipeline not ready
- Experimenting with time series methods

### When to use FEATURES mode:
- Need interpretable features
- Want to analyze feature importance
- Have computational resources for TSFresh
- Aiming for best possible accuracy
- Need faster inference time

### Performance Comparison:

| Aspect | RAW Mode | FEATURES Mode |
|--------|----------|---------------|
| Setup | ✅ Fast | ⏳ Slow (feature extraction) |
| Training | ⚡ Fast (~5-10 min) | ⚡ Fast (~5-10 min) |
| Inference | 🐢 Slower | ⚡⚡ Faster |
| Accuracy | ✅ Good (80-87%) | ✅ Better (83-90%) |
| Interpretability | ❌ Limited | ✅ High |
| Memory | 💾 Higher | 💾 Lower |

---

## 🔍 Troubleshooting

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

**Low accuracy (<75%)**
- Check data quality and label distribution
- Ensure balanced labels (should be ~50/50)
- Try different classifier (Arsenal, XGBoost)
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

## 📝 Next Steps

After successful Model 1 training:

1. **Review Results**: Check classification report and confusion matrix
2. **Test Performance**: Run `python test_model1.py`
3. **Compare Modes**: Try both raw and features modes
4. **Proceed to Model 2**: Train 5-class non-stationary classifier
5. **Build Pipeline**: Combine Model 1 + Model 2 for hierarchical classification

---

## 🎯 Hierarchical Pipeline

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

## 📚 References

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
- **Column names**: 'series_id' or 'id' for series identifier
- **Data columns**: 'data' or 'value' for time series values
- **Labels**: 'is_stationary' metadata (True/False)

### Batch Processing

For memory efficiency, data is loaded in batches:
- **BATCH_SIZE**: 10 files at a time
- Garbage collection after each batch
- Prevents memory overflow on large datasets

---

**Last Updated:** 2025-11-01

