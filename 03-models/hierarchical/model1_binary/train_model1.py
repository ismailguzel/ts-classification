"""
Model 1: Binary Classification (Stationary vs Unstationary)
============================================================

Uses sktime for time series classification.
This is the first level of the hierarchical model.

Input: Labeled time series data
Output: Binary classifier (Stationary=0, Unstationary=1)

Models to try:
1. TimeSeriesForestClassifier - Fast, good baseline
2. ROCKET (Random Convolutional Kernel Transform) - SOTA
3. InceptionTime - Deep learning approach
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import time

# sktime imports
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.kernel_based import RocketClassifier

print("="*70)
print("MODEL 1: BINARY CLASSIFICATION (Stationary vs Unstationary)")
print("="*70)

# ============================================================================
# 1. Load and Prepare Data
# ============================================================================

print("\n[1/6] Loading data...")

data_path = Path('../../../data/processed/labeled_test-15000.parquet')
raw_path = Path('../../../data/raw/test-15000')

# Load data (prefer raw parquet, derive binary labels directly from is_stationary)
if raw_path.exists():
    files = list(raw_path.rglob('*.parquet'))
    if not files:
        print(f"❌ Error: No parquet files under {raw_path}")
        exit(1)
    dfs = []
    for fp in files:
        try:
            df_part = pd.read_parquet(fp)
            if 'series_id' not in df_part.columns or 'time' not in df_part.columns or 'data' not in df_part.columns:
                continue
            if 'is_stationary' not in df_part.columns:
                print(f"❌ Missing 'is_stationary' in {fp}")
                exit(1)
            # create unique_series_id for grouping
            source_file = str(fp.relative_to(raw_path))
            df_part['unique_series_id'] = source_file + '_' + df_part['series_id'].astype(str)
            dfs.append(df_part)
        except Exception as e:
            print(f"⚠️  Skipping {fp}: {e}")
    if not dfs:
        print("❌ Error: No valid parquet chunks loaded from raw.")
        exit(1)
    df = pd.concat(dfs, ignore_index=True)
    print(f"✓ Loaded {len(df):,} data points from raw parquet")
elif data_path.exists():
    df = pd.read_parquet(data_path)
    print(f"✓ Loaded {len(df):,} data points from processed labels")
else:
    print(f"❌ Error: Neither raw nor processed data found.\n  Missing raw: {raw_path}\n  Missing processed: {data_path}")
    exit(1)

# Compute binary labels (Model 1) using is_stationary if available; fallback to model1_label
if 'is_stationary' in df.columns:
    df['model1_label'] = (df['is_stationary'] == 0).astype(int)
elif 'model1_label' in df.columns:
    pass
else:
    print("❌ Error: Neither 'is_stationary' nor 'model1_label' present to derive labels.")
    exit(1)

# Group by series
print("\n[2/6] Preparing time series data...")
series_list = []
labels = []

for series_id, group in df.groupby('unique_series_id'):
    # Get time series values
    ts_data = group.sort_values('time')['data'].values
    
    # Get label (same for all points in series)
    label = group['model1_label'].iloc[0]
    
    series_list.append(ts_data)
    labels.append(label)

print(f"✓ Prepared {len(series_list):,} time series")

# Check label distribution
unique_labels, counts = np.unique(labels, return_counts=True)
print("\nLabel distribution:")
for label, count in zip(unique_labels, counts):
    name = "Stationary" if label == 0 else "Unstationary"
    pct = 100 * count / len(labels)
    print(f"  {int(label)}: {name:15s} {count:6,} ({pct:.1f}%)")

# ============================================================================
# 2. Create sktime-compatible format
# ============================================================================

print("\n[3/6] Converting to sktime format...")

# sktime expects 3D array: (n_samples, n_features, n_timepoints)
# For univariate series: (n_samples, 1, n_timepoints)

# Pad/truncate to same length (sktime requirement)
max_length = max(len(s) for s in series_list)
min_length = min(len(s) for s in series_list)
print(f"Series length range: {min_length} - {max_length}")

# Use a fixed length (truncate long, pad short)
fixed_length = 1500  # Reasonable middle ground

def prepare_series(series, target_length=1500):
    """Pad or truncate series to fixed length."""
    if len(series) > target_length:
        # Truncate (take middle part to preserve structure)
        start = (len(series) - target_length) // 2
        return series[start:start + target_length]
    elif len(series) < target_length:
        # Pad with zeros
        pad_width = target_length - len(series)
        return np.pad(series, (0, pad_width), mode='constant', constant_values=0)
    else:
        return series

X = np.array([prepare_series(s, fixed_length) for s in series_list])
X = X.reshape(X.shape[0], 1, X.shape[1])  # (n_samples, 1, n_timepoints)
y = np.array(labels)

print(f"✓ X shape: {X.shape}")
print(f"✓ y shape: {y.shape}")

# ============================================================================
# 3. Train/Test Split
# ============================================================================

print("\n[4/6] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set: {len(X_train):,} samples")
print(f"Test set:  {len(X_test):,} samples")

# ============================================================================
# 4. Train Models
# ============================================================================

print("\n[5/6] Training models...")
print("-" * 70)

models = {}
results = {}

# Model 1: TimeSeriesForest (Fast baseline)
print("\n🌲 Training TimeSeriesForestClassifier...")
start_time = time.time()
tsf = TimeSeriesForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
tsf.fit(X_train, y_train)
train_time = time.time() - start_time

y_pred = tsf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
models['TimeSeriesForest'] = tsf
results['TimeSeriesForest'] = {
    'accuracy': acc,
    'train_time': train_time,
    'predictions': y_pred
}
print(f"  ✓ Accuracy: {acc:.4f} ({100*acc:.2f}%)")
print(f"  ✓ Training time: {train_time:.2f}s")

# Model 2: ROCKET (SOTA)
print("\n🚀 Training ROCKET Classifier...")
start_time = time.time()
rocket = RocketClassifier(num_kernels=1000, random_state=42, n_jobs=-1)
rocket.fit(X_train, y_train)
train_time = time.time() - start_time

y_pred = rocket.predict(X_test)
acc = accuracy_score(y_test, y_pred)
models['ROCKET'] = rocket
results['ROCKET'] = {
    'accuracy': acc,
    'train_time': train_time,
    'predictions': y_pred
}
print(f"  ✓ Accuracy: {acc:.4f} ({100*acc:.2f}%)")
print(f"  ✓ Training time: {train_time:.2f}s")

# ============================================================================
# 5. Evaluate and Compare
# ============================================================================

print("\n[6/6] Evaluation Results")
print("=" * 70)

for model_name, result in results.items():
    print(f"\n📊 {model_name}")
    print("-" * 70)
    print(f"Accuracy: {result['accuracy']:.4f} ({100*result['accuracy']:.2f}%)")
    print(f"Training Time: {result['train_time']:.2f}s")
    
    print("\nClassification Report:")
    print(classification_report(
        y_test, 
        result['predictions'],
        target_names=['Stationary', 'Unstationary'],
        digits=4
    ))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, result['predictions'])
    print(f"                 Predicted")
    print(f"                 Stat  Unstat")
    print(f"Actual Stat      {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Unstat    {cm[1,0]:4d}  {cm[1,1]:4d}")

# ============================================================================
# 6. Save Best Model
# ============================================================================

print("\n" + "=" * 70)
print("SAVING BEST MODEL")
print("=" * 70)

# Select best model based on accuracy
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = models[best_model_name]
best_accuracy = results[best_model_name]['accuracy']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.4f} ({100*best_accuracy:.2f}%)")

# Save model
model_dir = Path('saved_models')
model_dir.mkdir(exist_ok=True)

model_path = model_dir / 'model1_binary_classifier.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

print(f"\n✓ Model saved to: {model_path}")

# Save metadata
metadata = {
    'model_name': best_model_name,
    'accuracy': best_accuracy,
    'train_time': results[best_model_name]['train_time'],
    'n_train': len(X_train),
    'n_test': len(X_test),
    'fixed_length': fixed_length,
    'classes': ['Stationary', 'Unstationary'],
    'feature_shape': X.shape[1:],
}

metadata_path = model_dir / 'model1_metadata.pkl'
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)

print(f"✓ Metadata saved to: {metadata_path}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("✅ MODEL 1 TRAINING COMPLETE!")
print("=" * 70)
print(f"\nBest Model: {best_model_name}")
print(f"Accuracy: {100*best_accuracy:.2f}%")
print(f"Saved to: {model_path}")
print("\nNext steps:")
print("  1. Review classification report above")
print("  2. If accuracy is good (>90%), proceed to Model 2")
print("  3. Train Model 2 (5-class unstationary types)")
print("=" * 70)

