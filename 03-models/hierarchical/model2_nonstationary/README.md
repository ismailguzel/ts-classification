# Model 2: Non-Stationary 5-Class Classification

## 🎯 Objective

Classify **NON-STATIONARY** time series into 5 semantic categories:

0. **Trend** (deterministic_trends) - Linear, quadratic, exponential trends
1. **Volatility** (volatility) - ARCH, GARCH, EGARCH patterns
2. **Stochastic** (stochastic) - Random walk, ARIMA processes
3. **Anomaly** (anomalies) - Point and collective anomalies
4. **Structural Break** (structural_breaks) - Mean, variance, trend shifts

This is the second level of the hierarchical classification system.
Only operates on series classified as **non-stationary** by Model 1.

---

## 🔄 Training Modes

### Mode 1: RAW (Default) - sktime classifiers
Uses raw time series with specialized time series classifiers:
- ✅ No feature engineering needed
- ✅ Fast training
- ✅ Good baseline performance
- 🌲 **TimeSeriesForest**: Fast ensemble method
- 🚀 **ROCKET**: State-of-the-art (2000 kernels for 5-class)
- ⚡ **MiniROCKET**: 10x faster than ROCKET
- 🎯 **Arsenal**: ROCKET-based ensemble (2000 kernels)
- 🔍 **ShapeletTransform**: Pattern-based classification
- 🏆 **HIVECOTEV2**: Most powerful (very slow)

### Mode 2: FEATURES - sklearn classifiers
Uses TSFresh extracted features with traditional ML:
- ✅ Potentially higher accuracy
- ✅ More interpretable features
- ✅ Faster inference
- 🌲 **Random Forest**: Robust ensemble
- 🚀 **XGBoost**: Gradient boosting
- ⚡ **SVM (RBF)**: Non-linear kernel for multi-class

---

## 📊 Models Available

### RAW Mode (sktime)

#### 1. TimeSeriesForest
- **Type**: Interval-based ensemble
- **Speed**: Fast ⚡⚡⚡
- **Accuracy**: 80-85%
- **Use case**: Quick baseline

#### 2. ROCKET
- **Type**: Convolutional kernel transform
- **Kernels**: 2000 (optimized for 5-class)
- **Speed**: Fast ⚡⚡
- **Accuracy**: 85-88%
- **Use case**: Balanced performance

#### 3. MiniROCKET ⭐ Recommended
- **Type**: Faster ROCKET variant
- **Speed**: Very fast ⚡⚡⚡
- **Accuracy**: 84-87%
- **Use case**: Best speed/accuracy ratio

#### 4. Arsenal ⭐ Best Accuracy
- **Type**: ROCKET ensemble (2000 kernels)
- **Speed**: Moderate ⚡
- **Accuracy**: 86-90%
- **Use case**: Highest accuracy

#### 5. ShapeletTransform
- **Type**: Pattern-based
- **Speed**: Slow ⏱️
- **Accuracy**: 82-86%
- **Use case**: Interpretable patterns

#### 6. HIVECOTEV2
- **Type**: Hybrid ensemble
- **Speed**: Very slow 🐌
- **Accuracy**: 88-92%
- **Use case**: Research/benchmarking only

### FEATURES Mode (sklearn)

#### 1. Random Forest
- **Type**: Decision tree ensemble
- **Estimators**: 200
- **Speed**: Fast ⚡
- **Use case**: Robust baseline

#### 2. XGBoost
- **Type**: Gradient boosting
- **Estimators**: 200
- **Speed**: Fast ⚡⚡
- **Performance**: Excellent
- **Use case**: Best performance

#### 3. SVM (RBF)
- **Type**: Support vector machine
- **Kernel**: RBF (for non-linear separation)
- **Speed**: Moderate ⏱️
- **Use case**: Complex decision boundaries

---

## 🚀 Usage

### Training - RAW Mode (Default)

```bash
cd 03-models/hierarchical/model2_nonstationary

# Train all models (default)
python train_model2.py --mode raw

# Train specific classifier
python train_model2.py --mode raw --classifier minirocket  # Fastest
python train_model2.py --mode raw --classifier arsenal     # Best accuracy
python train_model2.py --mode raw --classifier rocket      # Balanced
python train_model2.py --mode raw --classifier tsf         # Quick baseline
python train_model2.py --mode raw --classifier shapelet    # Pattern-based
python train_model2.py --mode raw --classifier hivecote    # Research (very slow)
```

**Available classifiers:**
- `all` - Train all models (default)
- `tsf` - TimeSeriesForest only
- `rocket` - ROCKET only (2000 kernels)
- `minirocket` - MiniROCKET only ⭐ **Recommended for speed**
- `arsenal` - Arsenal only (2000 kernels) ⭐ **Recommended for accuracy**
- `shapelet` - ShapeletTransform only
- `hivecote` - HIVECOTEV2 only (very slow, for 5-class)

**Requirements:**
- Raw data: `data/raw/unified-test/`
- Time (5-class is harder): 
  - MiniROCKET: ~30 seconds
  - ROCKET: ~90 seconds
  - Arsenal: ~2 minutes
  - All models: ~10-15 minutes
  - HIVECOTEV2: ~2+ hours
- Output: `saved_models/model2_nonstationary_classifier.pkl`

### Training - FEATURES Mode

```bash
# First: Extract and select features (if not done)
cd ../../../02-preprocessing
python extract_features.py --input ../data/raw/unified-test --output ../data/features
python feature_selection.py --input ../data/features --output ../data/features/selected --target primary

# Then: Train with features
cd ../03-models/hierarchical/model2_nonstationary
python train_model2.py --mode features --features-path ../../../data/features/selected
```

**Requirements:**
- Features: `data/features/selected/features_primary_*.parquet`
- Time: ~10-15 minutes
- Output: `saved_models/model2_nonstationary_classifier.pkl`

### Testing

```bash
# Test with 100 samples (default)
python test_model2.py

# Test with more samples
python test_model2.py --n-samples 500
```

---

## 📈 Expected Performance

**Target Accuracy:** 80-92% (5-class is harder than binary)

### RAW Mode Results (~750 non-stationary test samples)

| Classifier | Accuracy | Training Time | Speed | Use Case |
|------------|----------|---------------|-------|----------|
| TimeSeriesForest | 80-85% | ~45s | ⚡⚡⚡ | Quick baseline |
| ROCKET | 85-88% | ~90s | ⚡⚡ | Balanced |
| **MiniROCKET** ⭐ | 84-87% | ~30s | ⚡⚡⚡ | **Best speed/accuracy** |
| **Arsenal** ⭐ | 86-90% | ~2min | ⚡ | **Best accuracy** |
| Shapelet | 82-86% | ~10min | ⏱️ | Interpretable |
| HIVECOTEV2 | 88-92% | ~2+hrs | 🐌 | Research only |

### FEATURES Mode Results (~750 non-stationary test samples)

| Classifier | Accuracy | Training Time | Use Case |
|------------|----------|---------------|----------|
| Random Forest | 85-88% | ~10s | Robust baseline |
| XGBoost | 88-92% | ~15s | Best performance |
| SVM (RBF) | 83-87% | ~20s | Non-linear patterns |

**Why harder than Model 1?**
- 5 classes instead of 2
- Some categories have overlapping characteristics
- Trends vs structural breaks can be similar
- Stochastic vs volatility patterns overlap

**Per-Class Performance (typical):**
- Trend: 85-92% (easiest to detect)
- Volatility: 80-88% (distinct GARCH patterns)
- Stochastic: 75-85% (overlaps with volatility)
- Anomaly: 85-90% (clear outliers)
- Structural Break: 80-88% (can overlap with trends)

**Recommendations:**
- 🏃 **Need speed?** → Use MiniROCKET
- 🎯 **Need accuracy?** → Use Arsenal or XGBoost (features)
- ⚖️ **Balanced?** → Use ROCKET or MiniROCKET
- 🔬 **Research?** → Compare all models

---

## 🔧 Technical Details

### RAW Mode - Data Preparation

1. **Filtering**: Only non-stationary series (is_stationary = False)
2. **Format**: Univariate time series
3. **Length**: Fixed to 1,500 points (pad/truncate)
4. **Shape**: (n_samples, 1, 1500) for sktime
5. **Split**: 80% train, 20% test (stratified)

### FEATURES Mode - Data Preparation

1. **Filtering**: Only non-stationary series
2. **Format**: Feature vectors from TSFresh
3. **Features**: 50-150 selected features for primary categories
4. **Scaling**: StandardScaler normalization
5. **Split**: 80% train, 20% test (stratified)

### Category Mapping

```python
CATEGORY_MAPPING = {
    'deterministic_trends': 0,  # Trend
    'volatility': 1,            # Volatility
    'stochastic': 2,            # Stochastic
    'point_anomalies': 3,       # Anomaly
    'collective_anomalies': 3,  # Anomaly (merged)
    'structural_breaks': 4      # Structural Break
}
```

### Model Selection

Script automatically:
1. Trains multiple models (depending on mode)
2. Compares accuracy on test set
3. Saves best performing model
4. Stores metadata (mode, params, scaler, mapping, etc.)

### Output Files

```
saved_models/
├── model2_nonstationary_classifier.pkl    # Best trained model
└── model2_metadata.pkl                    # Training metadata
```

Metadata includes:
- Model name and type
- Training mode (raw/features)
- Accuracy metrics
- Class names and mapping
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
| Training | ⚡ Fast (~10-20 min) | ⚡ Fast (~10-15 min) |
| Inference | 🐢 Slower | ⚡⚡ Faster |
| Accuracy | ✅ Good (80-90%) | ✅ Better (85-92%) |
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
python feature_selection.py --target primary
```

**Error: No non-stationary series found**
- Check that data has non-stationary series (is_stationary = False)
- Verify primary_category field exists
- Ensure data generation completed successfully

**Low accuracy (<75%)**
- Check class distribution (should be balanced)
- Try different mode (raw vs features)
- Increase dataset size
- Tune hyperparameters

**Out of memory** (raw mode)
- Reduce fixed_length parameter
- Use features mode instead
- Process in smaller batches

---

## 🎯 Hierarchical Pipeline

Model 2 is part of a hierarchical system:

```
Input Time Series
      ↓
[Model 1: Binary]
Is Stationary?
      ↓
  YES → Stationary (stop)
      ↓
   NO → Continue to Model 2
      ↓
[Model 2: 5-Class] ← YOU ARE HERE
What type of non-stationary?
      ↓
  0: Trend
  1: Volatility
  2: Stochastic
  3: Anomaly
  4: Structural Break
      ↓
[Model 3: Sub-categories] (future)
Detailed classification within each category
```

### Building Hierarchical Pipeline:

```python
# Load both models
model1 = load_model('model1_binary_classifier.pkl')
model2 = load_model('model2_nonstationary_classifier.pkl')

# Step 1: Check if stationary
is_stationary = model1.predict(X)

# Step 2: If non-stationary, classify type
if not is_stationary:
    category = model2.predict(X)
    # Returns: 0=Trend, 1=Volatility, 2=Stochastic, 3=Anomaly, 4=Structural
```

---

## 📝 Next Steps

After successful Model 2 training:

1. **Review Results**: Check classification report and confusion matrix
2. **Test Performance**: Run test_model2.py
3. **Analyze Confusion**: Which classes are confused?
4. **Compare Modes**: Try both raw and features modes
5. **Build Pipeline**: Combine Model 1 + Model 2 for hierarchical classification
6. **Proceed to Model 3**: Train sub-category classifiers (optional)

---

## 📚 References

- **sktime**: https://www.sktime.net/
- **ROCKET**: Dempster et al., 2020
- **TSFresh**: https://tsfresh.readthedocs.io/
- **Hierarchical Classification**: Survey papers on multi-level classification

---

## 🆘 Common Issues

**Confusion between Trend and Structural Break:**
- Both involve changes over time
- Structural break is sudden, trend is gradual
- May need more sophisticated features

**Confusion between Stochastic and Volatility:**
- Both have random components
- Volatility focuses on variance patterns (GARCH)
- Stochastic focuses on mean patterns (random walk)

**Poor Anomaly Detection:**
- Anomalies are point/collective events
- May be rare in test set
- Consider separate anomaly detection methods

**Solutions:**
1. Use features mode for better discrimination
2. Analyze feature importance to understand separation
3. Consider ensemble methods
4. Increase training data for confused classes
