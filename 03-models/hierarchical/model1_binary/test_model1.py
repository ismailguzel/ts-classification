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
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Parse arguments
parser = argparse.ArgumentParser(description='Test Model 1: Binary Classification')
parser.add_argument('--n-samples', type=int, default=100,
                    help='Number of samples to test (default: 100)')
parser.add_argument('--output-dir', type=str, default=None,
                    help='Directory to store detailed predictions and metrics')

args = parser.parse_args()

OUTPUT_DIR = Path(args.output_dir).resolve() if args.output_dir else None
if OUTPUT_DIR:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Stationary', 'Non-Stationary']


def safe_predict_proba(model, X):
    if hasattr(model, 'predict_proba'):
        try:
            return model.predict_proba(X)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"⚠️  predict_proba unavailable: {exc}")
            return None
    return None

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
    
    # Detect id column name
    id_col = 'id' if 'id' in df.columns else 'series_id'
    
    # Take first N series
    test_series = []
    test_labels = []
    sample_ids = []
    
    for i, series_id in enumerate(df[id_col].unique()):
        if i >= args.n_samples:
            break
        series_data = df[df[id_col] == series_id].sort_values('time')
        
        # Check for data or value column
        if 'data' in series_data.columns:
            ts_data = series_data['data'].values
        elif 'value' in series_data.columns:
            ts_data = series_data['value'].values
        else:
            ts_data = series_data.iloc[:, 2].values  # Assuming third column is data
        
        label = 0 if series_data['is_stationary'].iloc[0] else 1
        
        test_series.append(ts_data)
        test_labels.append(label)
        sample_ids.append(series_id)
    
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
    sample_ids = np.array(sample_ids)

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
    sample_ids = X_df.index.to_numpy()

print(f"✓ Prepared {len(X_test)} test samples")

# Predict
print("\n[3/3] Making predictions...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
y_proba = safe_predict_proba(model, X_test)

print(f"\n✓ Test Accuracy: {100*accuracy:.2f}%")

print("\nClassification Report:")
report_text = classification_report(
    y_test, y_pred,
    target_names=CLASS_NAMES,
    digits=4
)
print(report_text)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                   Predicted")
print(f"                   Stat    Non-Stat")
print(f"Actual Stat        {cm[0,0]:4d}    {cm[0,1]:4d}")
print(f"       Non-Stat    {cm[1,0]:4d}    {cm[1,1]:4d}")

report_dict = classification_report(
    y_test,
    y_pred,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

misclassified_mask = (y_pred != y_test)
misclassified_count = int(np.sum(misclassified_mask))
if misclassified_count > 0:
    print(f"\n⚠️  Misclassified samples: {misclassified_count}")
    preview_ids = sample_ids[misclassified_mask][:5]
    print(f"    Examples: {preview_ids}")
else:
    print("\nNo misclassifications detected in this subset.")

predictions_df = pd.DataFrame({
    'sample_id': sample_ids,
    'true_label': y_test,
    'predicted_label': y_pred
})

if y_proba is not None:
    proba_array = np.asarray(y_proba)
    for idx, cls_name in enumerate(CLASS_NAMES):
        col_name = f"prob_{cls_name.lower().replace(' ', '_')}"
        predictions_df[col_name] = proba_array[:, idx]

misclassified_df = predictions_df[predictions_df['true_label'] != predictions_df['predicted_label']].copy()

results_payload = {
    'timestamp_utc': datetime.utcnow().isoformat(timespec='seconds'),
    'mode': mode,
    'model_name': metadata['model_name'],
    'n_samples': int(len(X_test)),
    'test_accuracy': float(accuracy),
    'classification_report': report_dict,
    'confusion_matrix': cm.tolist(),
    'misclassified_count': misclassified_count,
    'probability_available': y_proba is not None
}

artifacts = {}

if OUTPUT_DIR:
    metrics_path = OUTPUT_DIR / f"model1_test_metrics_{mode}.json"
    artifacts['metrics_json'] = str(metrics_path)

    predictions_path = OUTPUT_DIR / f"model1_test_predictions_{mode}.csv"
    predictions_df.to_csv(predictions_path, index=False)
    artifacts['predictions_csv'] = str(predictions_path)

    if not misclassified_df.empty:
        misclassified_path = OUTPUT_DIR / f"model1_test_misclassified_{mode}.csv"
        misclassified_df.to_csv(misclassified_path, index=False)
        artifacts['misclassified_csv'] = str(misclassified_path)

    results_payload['artifacts'] = artifacts
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results_payload, f, indent=2)
else:
    results_payload['artifacts'] = artifacts

print("="*80)
print("✅ TEST COMPLETE!")
print("="*80)
print(f"\nTested {len(X_test)} samples")
print(f"Mode: {mode.upper()}")
print(f"Model: {metadata['model_name']}")
print(f"Accuracy: {100*accuracy:.2f}%")
print("="*80)

