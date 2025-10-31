# Model 1: Binary Classification

## 🎯 Objective

Classify time series as **Stationary (0)** or **Non-Stationary (1)**.

This is the first level of the hierarchical classification system.

---

## � Training Modes

### Mode 1: RAW (Default) - sktime classifiers
Uses raw time series with specialized time series classifiers:
- ✅ No feature engineering needed
- ✅ Fast training
- ✅ Good baseline performance
- ⚡ **TimeSeriesForest**: Fast ensemble method
- 🚀 **ROCKET**: State-of-the-art, very fast

### Mode 2: FEATURES - sklearn classifiers
Uses TSFresh extracted features with traditional ML:
- ✅ Potentially higher accuracy
- ✅ More interpretable features
- ✅ Faster inference
- 🌲 **Random Forest**: Robust ensemble
- 🚀 **XGBoost**: Gradient boosting
- ⚡ **SVM**: Linear classifier

---

## 📊 Models Available

### RAW Mode (sktime)

#### 1. TimeSeriesForest
- **Type**: Interval-based ensemble
- **Speed**: Fast ⚡
- **Use case**: Quick baseline

#### 2. ROCKET
- **Type**: Convolutional kernel transform
- **Speed**: Very fast ⚡⚡⚡
- **Performance**: State-of-the-art
- **Use case**: Production model

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

# Train with raw time series (sktime)
python train_model1.py --mode raw
```

**Requirements:**
- Raw data: `data/raw/unified-test/`
- Time: ~5-15 minutes
- Output: `saved_models/model1_binary_classifier.pkl`

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
- Features: `data/features/selected/features_binary_*.parquet`
- Time: ~5-10 minutes
- Output: `saved_models/model1_binary_classifier.pkl`

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

**Target Accuracy:** >95%

### RAW Mode Results (15K dataset):
- TimeSeriesForest: ~93-95%
- ROCKET: ~95-97%

### FEATURES Mode Results (15K dataset):
- Random Forest: ~95-97%
- XGBoost: ~96-98%
- SVM: ~94-96%

**Why high accuracy?**
Stationary vs non-stationary is a fundamental distinction that's relatively easy to detect from time series properties.

---

## 🔧 Technical Details

### RAW Mode - Data Preparation

1. **Format**: Univariate time series
2. **Length**: Fixed to 1,500 points (pad/truncate)
3. **Shape**: (n_samples, 1, 1500) for sktime
4. **Split**: 80% train, 20% test (stratified)

### FEATURES Mode - Data Preparation

1. **Format**: Feature vectors from TSFresh
2. **Features**: 50-150 selected features
3. **Scaling**: StandardScaler normalization
4. **Split**: 80% train, 20% test (stratified)

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
- Scaler (if features mode)

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
| Training | ⚡ Fast | ⚡ Fast |
| Inference | 🐢 Slower | ⚡⚡ Faster |
| Accuracy | ✅ Good (93-97%) | ✅ Better (95-98%) |
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

**Low accuracy (<90%)**
- Check data quality
- Ensure balanced labels
- Try different mode
- Increase dataset size

**Out of memory** (raw mode)
- Reduce fixed_length parameter
- Use features mode instead
- Process in smaller batches

---

## 📝 Next Steps

After successful Model 1 training:

1. **Review Results**: Check classification report
2. **Test Performance**: Run test_model1.py
3. **Compare Modes**: Try both raw and features modes
4. **Proceed to Model 2**: Train primary category classifier
5. **Ensemble**: Combine with other models for hierarchical classification

---

## 📚 References

- **sktime**: https://www.sktime.net/
- **ROCKET**: Dempster et al., 2020
- **TSFresh**: https://tsfresh.readthedocs.io/
- **Time Series Classification**: Bagnall et al., 2017
└── model1_metadata.pkl              # Model info
```

---

## 📝 Notes

- **Fixed Length**: All series are padded/truncated to 1,500 points
  - Preserves most information (min length ~1,000)
  - Enables fast training
  - Sktime requirement

- **Feature Engineering**: Not needed!
  - Models extract features automatically
  - ROCKET uses random convolutions
  - TSF uses intervals

- **Interpretability**: 
  - TSF: Shows important intervals
  - ROCKET: Black box (but fast!)

---

## 🐛 Troubleshooting

### "Data file not found"
```bash
cd 02-preprocessing
python3 create_labels.py
```

### "sktime not installed"
```bash
pip install sktime scikit-learn
```

### Low accuracy (<90%)
- Check data quality
- Try different models
- Adjust hyperparameters
- Increase training data

---

## 🎯 Next Steps

After Model 1 is trained:

1. ✅ Verify accuracy >95%
2. ⏭️ Train Model 2 (5-class unstationary types)
3. ⏭️ Train Model 3a-e (subtypes)
4. ⏭️ Build cascade pipeline
5. ⏭️ Evaluate end-to-end performance

---

**Last Updated:** 2025-10-30

