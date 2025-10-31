"""
Test Model 1: Binary Classification
====================================

Quick test script to verify trained model.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

print("="*70)
print("MODEL 1 TEST - Binary Classification")
print("="*70)

# Load model
model_path = Path('saved_models/model1_binary_classifier.pkl')
metadata_path = Path('saved_models/model1_metadata.pkl')

if not model_path.exists():
    print("❌ Model not found! Please train first:")
    print("   python3 train_model1.py")
    exit(1)

print("\n[1/3] Loading model...")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(metadata_path, 'rb') as f:
    metadata = pickle.load(f)

print(f"✓ Loaded model: {metadata['model_name']}")
print(f"  Training accuracy: {100*metadata['accuracy']:.2f}%")
print(f"  Feature shape: {metadata['feature_shape']}")

# Load test data (small sample)
print("\n[2/3] Loading test data...")
data_path = Path('../../../data/processed/labeled_test-15000.parquet')
df = pd.read_parquet(data_path)

# Take first 100 series as quick test
test_series = []
test_labels = []

for i, (series_id, group) in enumerate(df.groupby('unique_series_id')):
    if i >= 100:
        break
    ts_data = group.sort_values('time')['data'].values
    label = group['model1_label'].iloc[0]
    test_series.append(ts_data)
    test_labels.append(label)

# Prepare data
def prepare_series(series, target_length=1500):
    if len(series) > target_length:
        start = (len(series) - target_length) // 2
        return series[start:start + target_length]
    elif len(series) < target_length:
        pad_width = target_length - len(series)
        return np.pad(series, (0, pad_width), mode='constant')
    return series

X_test = np.array([prepare_series(s) for s in test_series])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
y_test = np.array(test_labels)

print(f"✓ Prepared {len(X_test)} test samples")

# Predict
print("\n[3/3] Making predictions...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✓ Test Accuracy: {100*accuracy:.2f}%")
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['Stationary', 'Unstationary']
))

print("="*70)
print("✅ TEST COMPLETE!")
print("="*70)

