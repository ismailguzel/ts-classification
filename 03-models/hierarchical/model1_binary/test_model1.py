"""
Test Model 1: Binary Classification
====================================

Quick test script to verify trained model.

Usage:
    python test_model1.py [--n-samples 100]
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
import warnings
warnings.filterwarnings('ignore')

# Parse arguments
parser = argparse.ArgumentParser(description='Test Model 1: Binary Classification')
parser.add_argument('--n-samples', type=int, default=100,
                    help='Number of samples to test (default: 100)')

args = parser.parse_args()

print("="*80)
print("MODEL 1 TEST - Binary Classification")
print("="*80)

# Load model
model_path = Path('saved_models/model1_binary_classifier.pkl')
metadata_path = Path('saved_models/model1_metadata.pkl')

if not model_path.exists():
    print("❌ Model not found! Please train first:")
    print("   python train_model1.py")
    exit(1)

print("\n[1/3] Loading model...")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(metadata_path, 'rb') as f:
    metadata = pickle.load(f)

print(f"✓ Loaded model: {metadata['model_name']}")
print(f"  Mode: {metadata['mode']}")
print(f"  Training accuracy: {100*metadata['accuracy']:.2f}%")
if metadata['mode'] == 'raw':
    print(f"  Feature shape: {metadata['feature_shape']}")
else:
    print(f"  Number of features: {metadata['n_features']}")

# Load test data
print("\n[2/3] Loading test data...")

mode = metadata['mode']

if mode == 'raw':
    # RAW MODE: Load time series
    data_path = Path('../../../data/raw/unified-test')
    
    if not data_path.exists():
        print(f"❌ Error: Test data not found: {data_path}")
        exit(1)
    
    # Search for parquet files recursively (they're in subdirectories by category)
    files = list(data_path.rglob('*.parquet'))
    if not files:
        print(f"❌ Error: No parquet files found in: {data_path}")
        print(f"    Searched recursively in all subdirectories")
        exit(1)
    
    print(f"Found {len(files)} parquet files")
    
    # Load and prepare test series
    dfs = []
    for fp in files[:5]:  # Load first few files for quick test
        df_part = pd.read_parquet(fp)
        dfs.append(df_part)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Take first N series
    test_series = []
    test_labels = []
    
    for i, series_id in enumerate(df['id'].unique()):
        if i >= args.n_samples:
            break
        series_data = df[df['id'] == series_id].sort_values('time')
        ts_data = series_data['value'].values
        label = 0 if series_data['is_stationary'].iloc[0] else 1
        
        test_series.append(ts_data)
        test_labels.append(label)
    
    # Prepare data
    fixed_length = metadata.get('fixed_length', 1500)
    
    def prepare_series(series, target_length):
        if len(series) > target_length:
            start = (len(series) - target_length) // 2
            return series[start:start + target_length]
        elif len(series) < target_length:
            pad_width = target_length - len(series)
            return np.pad(series, (0, pad_width), mode='constant')
        return series
    
    X_test = np.array([prepare_series(s, fixed_length) for s in test_series])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    y_test = np.array(test_labels)

else:
    # FEATURES MODE: Load TSFresh features
    features_path = Path('../../../data/features/selected')
    features_file = features_path / 'features_binary_mutual_info.parquet'
    labels_file = features_path / 'labels_binary.parquet'
    
    if not features_file.exists():
        print(f"❌ Error: Features file not found: {features_file}")
        exit(1)
    
    # Load features and labels
    X_df = pd.read_parquet(features_file)
    labels_df = pd.read_parquet(labels_file)
    
    # Set index to id
    if 'id' in X_df.columns:
        X_df = X_df.set_index('id')
    
    # Take first N samples
    X_df = X_df.head(args.n_samples)
    labels_df = labels_df.head(args.n_samples)
    
    # Convert labels
    y_test = (labels_df['is_stationary'] == False).astype(int).values
    
    # Scale features using saved scaler
    scaler = metadata.get('scaler')
    if scaler is not None:
        X_test = scaler.transform(X_df.values)
    else:
        X_test = X_df.values

print(f"✓ Prepared {len(X_test)} test samples")

# Predict
print("\n[3/3] Making predictions...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✓ Test Accuracy: {100*accuracy:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['Stationary', 'Non-Stationary'],
    digits=4
))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                   Predicted")
print(f"                   Stat    Non-Stat")
print(f"Actual Stat        {cm[0,0]:4d}    {cm[0,1]:4d}")
print(f"       Non-Stat    {cm[1,0]:4d}    {cm[1,1]:4d}")

print("="*80)
print("✅ TEST COMPLETE!")
print("="*80)
print(f"\nTested {len(X_test)} samples")
print(f"Mode: {mode.upper()}")
print(f"Model: {metadata['model_name']}")
print(f"Accuracy: {100*accuracy:.2f}%")
print("="*80)

