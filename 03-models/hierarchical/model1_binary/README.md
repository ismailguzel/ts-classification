# Model 1: Binary Classification

## 🎯 Objective

Classify time series as **Stationary (0)** or **Unstationary (1)**.

This is the first level of the hierarchical classification system.

---

## 📊 Models Implemented

### 1. TimeSeriesForest
- **Type**: Interval-based ensemble
- **Speed**: Fast ⚡
- **Use case**: Baseline, quick training

### 2. ROCKET (Random Convolutional Kernel Transform)
- **Type**: Feature-based with linear classifier
- **Speed**: Very fast ⚡⚡⚡
- **Performance**: State-of-the-art
- **Use case**: Production model

### 3. InceptionTime (Optional)
- **Type**: Deep learning (CNN)
- **Speed**: Slower 🐢
- **Performance**: Excellent
- **Use case**: When accuracy is critical

---

## 🚀 Usage

### Training

```bash
cd 03-models/hierarchical/model1_binary
python3 train_model1.py
```

**Requirements:**
- Labeled data: `data/processed/labeled_test-15000.parquet`
- Time: ~5-15 minutes (depends on model)
- Output: `saved_models/model1_binary_classifier.pkl`

### Testing

```bash
python3 test_model1.py
```

Quick test on 100 samples to verify model works.

---

## 📈 Expected Performance

**Target Accuracy:** >95%

**Typical Results (15K dataset):**
- TimeSeriesForest: ~93-95%
- ROCKET: ~95-97%
- InceptionTime: ~96-98%

**Why high accuracy?**
Stationary vs Unstationary is a fundamental distinction that's relatively easy to detect from time series properties.

---

## 🔧 Technical Details

### Data Preparation

1. **Format**: Univariate time series
2. **Length**: Fixed to 1,500 points (pad/truncate)
3. **Shape**: (n_samples, 1, 1500)
4. **Split**: 80% train, 20% test

### Model Selection

Script automatically:
1. Trains multiple models
2. Compares accuracy
3. Saves best model
4. Stores metadata

### Output Files

```
saved_models/
├── model1_binary_classifier.pkl    # Trained model
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

