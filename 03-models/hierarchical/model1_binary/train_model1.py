"""
Model 1: Binary Classification (Stationary vs Unstationary)
============================================================

Two training modes:
1. RAW MODE (default): Uses raw time series with sktime classifiers
   - TimeSeriesForestClassifier
   - ROCKET (Random Convolutional Kernel Transform)
   - MiniROCKET (Faster version of ROCKET)
   - Arsenal (ROCKET-based ensemble)
   - ShapeletTransformClassifier (Pattern-based)
   - HIVECOTEV2 (Most powerful, slowest)
   
2. FEATURES MODE (optional): Uses TSFresh features with sklearn classifiers
   - Random Forest
   - XGBoost
   - SVM

Usage:
    # Raw time series (sktime)
    python train_model1.py --mode raw
    
    # TSFresh features (sklearn)
    python train_model1.py --mode features --features-path ../../../data/features/selected
    
    # Choose specific classifier
    python train_model1.py --mode raw --classifier rocket
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

# Parse arguments
parser = argparse.ArgumentParser(description='Train Model 1: Binary Classification')
parser.add_argument('--mode', type=str, default='raw', choices=['raw', 'features'],
                    help='Training mode: raw (sktime) or features (sklearn)')
parser.add_argument('--data-path', type=str, default='../../../data/raw/unified-test',
                    help='Path to raw time series data')
parser.add_argument('--features-path', type=str, default='../../../data/features/selected',
                    help='Path to TSFresh features (for features mode)')
parser.add_argument('--test-size', type=float, default=0.2,
                    help='Test set size (default: 0.2)')
parser.add_argument('--random-state', type=int, default=42,
                    help='Random state for reproducibility')
parser.add_argument('--classifier', type=str, default='all',
                    choices=['all', 'tsf', 'rocket', 'minirocket', 'arsenal', 'shapelet', 'hivecote'],
                    help='Specific classifier to train (default: all)')

args = parser.parse_args()

print("="*80)
print("MODEL 1: BINARY CLASSIFICATION (Stationary vs Non-Stationary)")
print("="*80)
print(f"Mode: {args.mode.upper()}")
print(f"Data path: {args.data_path}")
if args.mode == 'features':
    print(f"Features path: {args.features_path}")
print("="*80)

# ============================================================================
# 1. Load and Prepare Data
# ============================================================================

if args.mode == 'raw':
    # RAW MODE: Load time series from parquet files
    print("\n[1/6] Loading raw time series data...")
    
    raw_path = Path(args.data_path)
    if not raw_path.exists():
        print(f"❌ Error: Data path not found: {raw_path}")
        print(f"    Please generate data first:")
        print(f"    cd ../../../01-data-generation && python generate_test.py")
        exit(1)
    
    # Search for parquet files recursively (they're in subdirectories by category)
    files = list(raw_path.rglob('*.parquet'))
    if not files:
        print(f"❌ Error: No parquet files found in: {raw_path}")
        print(f"    Searched recursively in all subdirectories")
        print(f"    Expected structure: {raw_path}/stationary/, {raw_path}/deterministic_trend_*, etc.")
        exit(1)
    
    print(f"Found {len(files)} parquet files across categories")
    
    # Show category distribution
    categories = {}
    for fp in files:
        cat = fp.parent.name
        categories[cat] = categories.get(cat, 0) + 1
    
    print("Category distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} files")
    
    # Load files in batches (memory-efficient)
    print("\nLoading files in batches (memory-efficient)...")
    BATCH_SIZE = 10  # Process 10 files at a time
    all_batches = []
    
    import gc
    for batch_idx in range(0, len(files), BATCH_SIZE):
        batch_files = files[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        total_batches = (len(files) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"  Batch {batch_num}/{total_batches}: Loading {len(batch_files)} files...")
        
        # Load batch
        batch_dfs = []
        for fp in batch_files:
            df_part = pd.read_parquet(fp)
            batch_dfs.append(df_part)
        
        # Concatenate batch
        batch_df = pd.concat(batch_dfs, ignore_index=True)
        all_batches.append(batch_df)
        
        # Clean up batch DataFrames
        del batch_dfs
        gc.collect()
        
        print(f"  ✓ Batch {batch_num} loaded: {len(batch_df):,} rows")
    
    # Final concatenation
    print("\nCombining all batches...")
    df = pd.concat(all_batches, ignore_index=True)
    del all_batches
    gc.collect()
    
    print(f"✓ Loaded {len(df):,} data points from {len(files)} files")
    
    # Extract time series and labels
    print("\n[2/6] Preparing time series data...")
    series_list = []
    labels = []
    
    for series_id in df['id'].unique():
        series_data = df[df['id'] == series_id].sort_values('time')
        ts_data = series_data['value'].values
        
        # Get label from is_stationary (True=stationary=0, False=non-stationary=1)
        label = 0 if series_data['is_stationary'].iloc[0] else 1
        
        series_list.append(ts_data)
        labels.append(label)
    
    print(f"✓ Prepared {len(series_list):,} time series")

else:
    # FEATURES MODE: Load TSFresh features
    print("\n[1/6] Loading TSFresh features...")
    
    features_path = Path(args.features_path)
    features_file = features_path / 'features_binary_mutual_info.parquet'
    labels_file = features_path / 'labels_binary.parquet'
    
    if not features_file.exists():
        print(f"❌ Error: Features file not found: {features_file}")
        print(f"    Please extract features first:")
        print(f"    cd ../../../02-preprocessing")
        print(f"    python extract_features.py")
        print(f"    python feature_selection.py --target binary")
        exit(1)
    
    if not labels_file.exists():
        print(f"❌ Error: Labels file not found: {labels_file}")
        exit(1)
    
    # Load features and labels
    X_df = pd.read_parquet(features_file)
    labels_df = pd.read_parquet(labels_file)
    
    # Set index to id
    if 'id' in X_df.columns:
        X_df = X_df.set_index('id')
    
    # Convert labels (is_stationary: True=0, False=1)
    labels = (labels_df['is_stationary'] == False).astype(int).values
    
    print(f"✓ Loaded features: {X_df.shape}")
    print(f"✓ Number of features: {X_df.shape[1]}")
    print(f"✓ Number of samples: {X_df.shape[0]}")
    
    # Store for later use
    X_features = X_df.values
    series_list = None  # Not used in features mode

# Check label distribution
labels = np.array(labels)
unique_labels, counts = np.unique(labels, return_counts=True)
print("\nLabel distribution:")
for label, count in zip(unique_labels, counts):
    name = "Stationary" if label == 0 else "Non-Stationary"
    pct = 100 * count / len(labels)
    print(f"  {int(label)}: {name:15s} {count:6,} ({pct:.1f}%)")

# ============================================================================
# 2. Prepare Data for Training
# ============================================================================

if args.mode == 'raw':
    # Convert to sktime-compatible format
    print("\n[3/6] Converting to sktime format...")
    
    # Import sktime classifiers
    from sktime.classification.interval_based import TimeSeriesForestClassifier
    from sktime.classification.kernel_based import RocketClassifier
    from sktime.classification.kernel_based import Arsenal
    try:
        from sktime.classification.shapelet_based import ShapeletTransformClassifier
        has_shapelet = True
    except ImportError:
        has_shapelet = False
        print("⚠️  ShapeletTransformClassifier not available in this sktime version")
    
    try:
        from sktime.classification.hybrid import HIVECOTEV2
        has_hivecote = True
    except ImportError:
        has_hivecote = False
        print("⚠️  HIVECOTEV2 not available in this sktime version")
    
    # sktime expects 3D array: (n_samples, n_features, n_timepoints)
    # For univariate series: (n_samples, 1, n_timepoints)
    
    # Pad/truncate to same length
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
    y = labels
    
    print(f"✓ X shape: {X.shape}")
    print(f"✓ y shape: {y.shape}")

else:
    # Features mode: Use TSFresh features directly
    print("\n[3/6] Preparing feature matrix...")
    
    # Import sklearn classifiers
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    try:
        from xgboost import XGBClassifier
        has_xgboost = True
    except ImportError:
        has_xgboost = False
        print("⚠️  XGBoost not installed, skipping XGBoost model")
    
    from sklearn.preprocessing import StandardScaler
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X_features)
    y = labels
    
    print(f"✓ X shape: {X.shape}")
    print(f"✓ y shape: {y.shape}")
    print(f"✓ Features scaled with StandardScaler")

# ============================================================================
# 3. Train/Test Split
# ============================================================================

print("\n[4/6] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
)

print(f"Train set: {len(X_train):,} samples")
print(f"Test set:  {len(X_test):,} samples")

# ============================================================================
# 4. Train Models
# ============================================================================

print("\n[5/6] Training models...")
print("-" * 80)

models = {}
results = {}

if args.mode == 'raw':
    # RAW MODE: Train sktime classifiers
    
    # Model 1: TimeSeriesForest (Fast baseline)
    if args.classifier in ['all', 'tsf']:
        print("\n🌲 Training TimeSeriesForestClassifier...")
        start_time = time.time()
        tsf = TimeSeriesForestClassifier(n_estimators=100, random_state=args.random_state, n_jobs=110)
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
        
        # Memory cleanup
        del tsf, y_pred
        import gc
        gc.collect()
        print("  ✓ Memory cleaned")
    
    # Model 2: ROCKET (SOTA)
    if args.classifier in ['all', 'rocket']:
        print("\n🚀 Training ROCKET Classifier...")
        start_time = time.time()
        rocket = RocketClassifier(num_kernels=1000, random_state=args.random_state, n_jobs=110)
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
    
    # Model 3: MiniROCKET (Faster ROCKET)
    if args.classifier in ['all', 'minirocket']:
        print("\n⚡ Training MiniROCKET Classifier...")
        try:
            from sktime.classification.kernel_based import MiniRocketClassifier
            start_time = time.time()
            minirocket = MiniRocketClassifier(random_state=args.random_state, n_jobs=110)
            minirocket.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            y_pred = minirocket.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            models['MiniROCKET'] = minirocket
            results['MiniROCKET'] = {
                'accuracy': acc,
                'train_time': train_time,
                'predictions': y_pred
            }
            print(f"  ✓ Accuracy: {acc:.4f} ({100*acc:.2f}%)")
            print(f"  ✓ Training time: {train_time:.2f}s")
        except ImportError:
            print("  ⚠️  MiniROCKET not available in this sktime version")
    
    # Model 4: Arsenal (ROCKET ensemble)
    if args.classifier in ['all', 'arsenal']:
        print("\n🎯 Training Arsenal Classifier...")
        start_time = time.time()
        arsenal = Arsenal(num_kernels=1000, random_state=args.random_state, n_jobs=110)
        arsenal.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = arsenal.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        models['Arsenal'] = arsenal
        results['Arsenal'] = {
            'accuracy': acc,
            'train_time': train_time,
            'predictions': y_pred
        }
        print(f"  ✓ Accuracy: {acc:.4f} ({100*acc:.2f}%)")
        print(f"  ✓ Training time: {train_time:.2f}s")
    
    # Model 5: ShapeletTransform (Pattern-based)
    if args.classifier in ['all', 'shapelet'] and has_shapelet:
        print("\n🔍 Training ShapeletTransformClassifier...")
        print("  ⚠️  This may take longer...")
        start_time = time.time()
        shapelet = ShapeletTransformClassifier(
            n_shapelet_samples=200,
            max_shapelets=20,
            batch_size=100,
            random_state=args.random_state,
            n_jobs=110
        )
        shapelet.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = shapelet.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        models['Shapelet'] = shapelet
        results['Shapelet'] = {
            'accuracy': acc,
            'train_time': train_time,
            'predictions': y_pred
        }
        print(f"  ✓ Accuracy: {acc:.4f} ({100*acc:.2f}%)")
        print(f"  ✓ Training time: {train_time:.2f}s")
    
    # Model 6: HIVECOTEV2 (Most powerful, very slow)
    if args.classifier in ['all', 'hivecote'] and has_hivecote:
        print("\n🏆 Training HIVECOTEV2...")
        print("  ⚠️  WARNING: This is VERY SLOW (may take hours)!")
        print("  ⚠️  Consider using smaller dataset or skip this model")
        confirm = input("  Continue? [y/N]: ")
        if confirm.lower() == 'y':
            start_time = time.time()
            hivecote = HIVECOTEV2(random_state=args.random_state, n_jobs=110)
            hivecote.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            y_pred = hivecote.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            models['HIVECOTEV2'] = hivecote
            results['HIVECOTEV2'] = {
                'accuracy': acc,
                'train_time': train_time,
                'predictions': y_pred
            }
            print(f"  ✓ Accuracy: {acc:.4f} ({100*acc:.2f}%)")
            print(f"  ✓ Training time: {train_time:.2f}s")
        else:
            print("  Skipped HIVECOTEV2")

else:
    # FEATURES MODE: Train sklearn classifiers
    
    # Model 1: Random Forest
    print("\n🌲 Training Random Forest...")
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=200, random_state=args.random_state, n_jobs=110)
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    models['RandomForest'] = rf
    results['RandomForest'] = {
        'accuracy': acc,
        'train_time': train_time,
        'predictions': y_pred
    }
    print(f"  ✓ Accuracy: {acc:.4f} ({100*acc:.2f}%)")
    print(f"  ✓ Training time: {train_time:.2f}s")
    
    # Model 2: XGBoost (if available)
    if has_xgboost:
        print("\n🚀 Training XGBoost...")
        start_time = time.time()
        xgb = XGBClassifier(n_estimators=200, random_state=args.random_state, n_jobs=110, 
                           eval_metric='logloss')
        xgb.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = xgb.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        models['XGBoost'] = xgb
        results['XGBoost'] = {
            'accuracy': acc,
            'train_time': train_time,
            'predictions': y_pred
        }
        print(f"  ✓ Accuracy: {acc:.4f} ({100*acc:.2f}%)")
        print(f"  ✓ Training time: {train_time:.2f}s")
    
    # Model 3: SVM (fast linear kernel for large feature sets)
    print("\n⚡ Training SVM (Linear)...")
    start_time = time.time()
    svm = SVC(kernel='linear', random_state=args.random_state)
    svm.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    models['SVM_Linear'] = svm
    results['SVM_Linear'] = {
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
print("=" * 80)

for model_name, result in results.items():
    print(f"\n📊 {model_name}")
    print("-" * 80)
    print(f"Accuracy: {result['accuracy']:.4f} ({100*result['accuracy']:.2f}%)")
    print(f"Training Time: {result['train_time']:.2f}s")
    
    print("\nClassification Report:")
    print(classification_report(
        y_test, 
        result['predictions'],
        target_names=['Stationary', 'Non-Stationary'],
        digits=4
    ))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, result['predictions'])
    print(f"                   Predicted")
    print(f"                   Stat    Non-Stat")
    print(f"Actual Stat        {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"       Non-Stat    {cm[1,0]:4d}    {cm[1,1]:4d}")

# ============================================================================
# 6. Save Best Model
# ============================================================================

print("\n" + "=" * 80)
print("SAVING BEST MODEL")
print("=" * 80)

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
    'mode': args.mode,
    'accuracy': best_accuracy,
    'train_time': results[best_model_name]['train_time'],
    'n_train': len(X_train),
    'n_test': len(X_test),
    'classes': ['Stationary', 'Non-Stationary'],
}

# Add mode-specific metadata
if args.mode == 'raw':
    metadata['fixed_length'] = fixed_length
    metadata['feature_shape'] = X.shape[1:]
else:
    metadata['n_features'] = X.shape[1]
    metadata['scaler'] = scaler

metadata_path = model_dir / 'model1_metadata.pkl'
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)

print(f"✓ Metadata saved to: {metadata_path}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("✅ MODEL 1 TRAINING COMPLETE!")
print("=" * 80)
print(f"\nMode: {args.mode.upper()}")
print(f"Best Model: {best_model_name}")
print(f"Accuracy: {100*best_accuracy:.2f}%")
print(f"Saved to: {model_path}")
print("\nNext steps:")
print("  1. Review classification report above")
print("  2. Test model: python test_model1.py")
print("  3. If accuracy is good (>90%), proceed to Model 2 (primary categories)")
if args.mode == 'raw':
    print("\nTip: Try features mode for potentially better performance:")
    print("  python train_model1.py --mode features --features-path ../../../data/features/selected")
print("=" * 80)

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

