"""
Model 2: Non-Stationary 5-Class Classification
==============================================

Classifies NON-STATIONARY time series into 5 semantic categories:
0. Trend (deterministic_trends)
1. Volatility (volatility)
2. Stochastic (stochastic)
3. Anomaly (point_anomalies + collective_anomalies)
4. Structural Break (structural_breaks)

This is the second level of the hierarchical classification system.
Only operates on series classified as non-stationary by Model 1.

Two training modes:
1. RAW MODE (default): Uses raw time series with sktime classifiers
   - TimeSeriesForestClassifier
   - ROCKET (2000 kernels for 5-class)
   - Arsenal (ROCKET-based ensemble)
   - ShapeletTransformClassifier (Pattern-based)

2. FEATURES MODE (optional): Uses TSFresh features with sklearn classifiers

Usage:
    # Raw time series (sktime)
    python train_model2.py --mode raw
    
    # TSFresh features (sklearn)
    python train_model2.py --mode features --features-path ../../../data/features/selected
    
    # Choose specific classifier
    python train_model2.py --mode raw --classifier rocket
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
parser = argparse.ArgumentParser(description='Train Model 2: Non-Stationary 5-Class Classification')
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
                    choices=['all', 'tsf', 'rocket', 'arsenal', 'shapelet'],
                    help='Specific classifier to train (default: all)')

args = parser.parse_args()

# Global n_jobs setting for all models
N_JOBS = 110  # Use all available cores

print("="*80)
print("MODEL 2: NON-STATIONARY 5-CLASS CLASSIFICATION")
print("="*80)
print(f"Mode: {args.mode.upper()}")
print(f"Data path: {args.data_path}")
if args.mode == 'features':
    print(f"Features path: {args.features_path}")
print(f"n_jobs: {N_JOBS}")
print("="*80)

# Define category mapping
CATEGORY_MAPPING = {
    'trend': 0,              # Trend (deterministic_trends in folder names)
    'volatility': 1,         # Volatility
    'stochastic': 2,         # Stochastic
    'anomaly': 3,            # Anomaly (point & collective anomalies)
    'structural_break': 4    # Structural Break (mean/variance/trend shifts)
}

CLASS_NAMES = ['Trend', 'Volatility', 'Stochastic', 'Anomaly', 'Structural Break']

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
    all_files = list(raw_path.rglob('*.parquet'))
    if not all_files:
        print(f"❌ Error: No parquet files found in: {raw_path}")
        print(f"    Searched recursively in all subdirectories")
        print(f"    Expected structure: {raw_path}/stationary/, {raw_path}/deterministic_trend_*, etc.")
        exit(1)
    
    # Filter out stationary files (Model 2 only processes non-stationary)
    files = [f for f in all_files if 'stationary' not in str(f.parent).lower()]
    
    print(f"Found {len(all_files)} total parquet files")
    print(f"Filtered to {len(files)} non-stationary files (excluded stationary)")
    
    if len(files) == 0:
        print("❌ Error: No non-stationary files found!")
        exit(1)
    
    # Show category distribution (non-stationary only)
    categories = {}
    for fp in files:
        # Get the primary category (parent or grandparent folder)
        # For nested structure like "deterministic_trend_quadratic/up/ar"
        # we want "deterministic_trend_quadratic", not "ar"
        parts = fp.parts
        raw_idx = parts.index('unified-test') if 'unified-test' in parts else -1
        
        if raw_idx != -1 and raw_idx + 1 < len(parts):
            primary_cat = parts[raw_idx + 1]  # First folder after unified-test
            categories[primary_cat] = categories.get(primary_cat, 0) + 1
    
    print("\nNon-stationary category distribution (by primary category):")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} files")
    
    # Load and extract time series INCREMENTALLY (memory-efficient)
    print("\nLoading and extracting time series incrementally...")
    print("This avoids loading entire dataset into memory at once.")
    
    series_list = []
    labels = []
    total_series = 0
    
    import gc
    BATCH_SIZE = 10  # Process 10 files at a time
    
    for batch_idx in range(0, len(files), BATCH_SIZE):
        batch_files = files[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        total_batches = (len(files) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"  Batch {batch_num}/{total_batches}: Processing {len(batch_files)} files...")
        
        # Process each file in the batch
        for fp in batch_files:
            try:
                df_part = pd.read_parquet(fp)
                
                # Expect series_id and data columns (standard format from generation)
                if 'series_id' not in df_part.columns:
                    raise ValueError(f"Column 'series_id' not found in {fp}")
                if 'data' not in df_part.columns:
                    raise ValueError(f"Column 'data' not found in {fp}")
                
                # Filter for non-stationary only
                df_part = df_part[df_part['is_stationary'] == False].copy()
                
                if len(df_part) == 0:
                    continue
                
                # Process each series in the file
                for series_id in df_part['series_id'].unique():
                    series_data = df_part[df_part['series_id'] == series_id].sort_values('time')
                    ts_data = series_data['data'].values
                    
                    # Get primary category and map to Model 2 label
                    primary_cat = series_data['primary_category'].iloc[0]
                    
                    if primary_cat not in CATEGORY_MAPPING:
                        continue  # Skip unknown categories
                    
                    label = CATEGORY_MAPPING[primary_cat]
                    
                    series_list.append(ts_data)
                    labels.append(label)
                    total_series += 1
                
                # Clean up immediately
                del df_part
                
            except Exception as e:
                print(f"  ⚠️  Error processing {fp.name}: {e}")
                continue
        
        gc.collect()
        print(f"  ✓ Batch {batch_num}: Extracted series (Total: {total_series})")
    
    if len(series_list) == 0:
        print("❌ Error: No non-stationary series found!")
        exit(1)
    
    print(f"\n✓ Prepared {len(series_list):,} time series without loading full dataset")
    
    # Display label distribution
    print("\n[2/6] Analyzing label distribution...")
    unique_labels, counts = np.unique(labels, return_counts=True)

else:
    # FEATURES MODE: Load TSFresh features
    print("\n[1/6] Loading TSFresh features...")
    
    features_path = Path(args.features_path)
    features_file = features_path / 'features_primary_mutual_info.parquet'
    labels_file = features_path / 'labels_primary.parquet'
    
    if not features_file.exists():
        print(f"❌ Error: Features file not found: {features_file}")
        print(f"    Please extract features first:")
        print(f"    cd ../../../02-preprocessing")
        print(f"    python extract_features.py")
        print(f"    python feature_selection.py --target primary")
        exit(1)
    
    if not labels_file.exists():
        print(f"❌ Error: Labels file not found: {labels_file}")
        exit(1)
    
    # Load features and labels
    X_df = pd.read_parquet(features_file)
    labels_df = pd.read_parquet(labels_file)
    
    # TSFresh outputs features with 'id' as index, labels have 'id' column
    if 'id' in X_df.columns:
        X_df = X_df.set_index('id')
    
    # Filter only NON-STATIONARY series
    nonstat_mask = labels_df['is_stationary'] == False
    X_df = X_df[nonstat_mask]
    labels_df = labels_df[nonstat_mask]
    
    print(f"✓ Filtered to {len(X_df):,} non-stationary series")
    
    if len(X_df) == 0:
        print("❌ Error: No non-stationary series found!")
        exit(1)
    
    # Map primary categories to Model 2 labels
    labels = labels_df['primary_category'].map(CATEGORY_MAPPING).values
    
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
    pct = 100 * count / len(labels)
    print(f"  {int(label)}: {CLASS_NAMES[label]:20s} {count:6,} ({pct:.1f}%)")

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
    
    # Pad/truncate to same length
    max_length = max(len(s) for s in series_list)
    min_length = min(len(s) for s in series_list)
    print(f"Series length range: {min_length} - {max_length}")
    
    # Use a fixed length
    fixed_length = 1500
    
    def prepare_series(series, target_length=1500):
        """Pad or truncate series to fixed length."""
        if len(series) > target_length:
            start = (len(series) - target_length) // 2
            return series[start:start + target_length]
        elif len(series) < target_length:
            pad_width = target_length - len(series)
            return np.pad(series, (0, pad_width), mode='constant', constant_values=0)
        else:
            return series
    
    X = np.array([prepare_series(s, fixed_length) for s in series_list])
    
    # Check for and handle NaN values
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"⚠️  Found {nan_count} NaN values in data, replacing with 0")
        X = np.nan_to_num(X, nan=0.0)
    
    # Check for inf values
    inf_count = np.isinf(X).sum()
    if inf_count > 0:
        print(f"⚠️  Found {inf_count} inf values in data, replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
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
    
    # Model 1: TimeSeriesForest
    if args.classifier in ['all', 'tsf']:
        print("\n🌲 Training TimeSeriesForestClassifier...")
        start_time = time.time()
        tsf = TimeSeriesForestClassifier(n_estimators=100, random_state=args.random_state, n_jobs=N_JOBS)
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
    
    # Model 2: ROCKET (2000 kernels for 5-class)
    if args.classifier in ['all', 'rocket']:
        print("\n🚀 Training ROCKET Classifier...")
        try:
            start_time = time.time()
            rocket = RocketClassifier(num_kernels=2000, random_state=args.random_state, n_jobs=N_JOBS)
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
        except (AttributeError, ImportError) as e:
            print(f"  ⚠️  ROCKET not available: {str(e)[:100]}")
            print(f"  ⚠️  This may be due to NumPy 2.0 incompatibility. Consider downgrading to numpy<2.0")
    
    # Model 3: Arsenal (ROCKET ensemble)
    if args.classifier in ['all', 'arsenal']:
        print("\n🎯 Training Arsenal Classifier...")
        try:
            start_time = time.time()
            arsenal = Arsenal(num_kernels=2000, random_state=args.random_state, n_jobs=N_JOBS)
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
        except (ImportError, AttributeError) as e:
            print(f"  ⚠️  Arsenal not available: {str(e)[:100]}")
            print(f"  ⚠️  This may be due to NumPy 2.0 incompatibility. Consider downgrading to numpy<2.0")
    
    # Model 5: ShapeletTransform (Pattern-based)
    if args.classifier in ['all', 'shapelet'] and has_shapelet:
        print("\n🔍 Training ShapeletTransformClassifier...")
        print("  ⚠️  This may take longer for 5-class problem...")
        start_time = time.time()
        shapelet = ShapeletTransformClassifier(
            n_shapelet_samples=200,
            max_shapelets=20,
            batch_size=100,
            random_state=args.random_state,
            n_jobs=N_JOBS
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

else:
    # FEATURES MODE: Train sklearn classifiers
    
    # Model 1: Random Forest
    print("\n🌲 Training Random Forest...")
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=200, random_state=args.random_state, n_jobs=N_JOBS)
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
        xgb = XGBClassifier(n_estimators=200, random_state=args.random_state, n_jobs=N_JOBS, 
                           eval_metric='mlogloss')
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
    
    # Model 3: SVM (RBF kernel for multi-class)
    print("\n⚡ Training SVM (RBF)...")
    start_time = time.time()
    svm = SVC(kernel='rbf', random_state=args.random_state)
    svm.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    models['SVM_RBF'] = svm
    results['SVM_RBF'] = {
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
        target_names=CLASS_NAMES,
        digits=4
    ))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, result['predictions'])
    print(f"{'':20s} " + " ".join([f"{name[:8]:>8s}" for name in CLASS_NAMES]))
    for i, name in enumerate(CLASS_NAMES):
        row = " ".join([f"{cm[i,j]:8d}" for j in range(len(CLASS_NAMES))])
        print(f"{name[:20]:20s} {row}")

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

model_path = model_dir / 'model2_nonstationary_classifier.pkl'
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
    'n_classes': 5,
    'class_names': CLASS_NAMES,
    'category_mapping': CATEGORY_MAPPING,
}

# Add mode-specific metadata
if args.mode == 'raw':
    metadata['fixed_length'] = fixed_length
    metadata['feature_shape'] = X.shape[1:]
else:
    metadata['n_features'] = X.shape[1]
    metadata['scaler'] = scaler

metadata_path = model_dir / 'model2_metadata.pkl'
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)

print(f"✓ Metadata saved to: {metadata_path}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("✅ MODEL 2 TRAINING COMPLETE!")
print("=" * 80)
print(f"\nMode: {args.mode.upper()}")
print(f"Best Model: {best_model_name}")
print(f"Accuracy: {100*best_accuracy:.2f}%")
print(f"Saved to: {model_path}")
print("\nNext steps:")
print("  1. Review classification report and confusion matrix above")
print("  2. Test model: python test_model2.py")
print("  3. If accuracy is good (>80%), proceed to Model 3 (sub-categories)")
print("  4. Or build hierarchical pipeline combining Model 1 + Model 2")
if args.mode == 'raw':
    print("\nTip: Try features mode for potentially better performance:")
    print("  python train_model2.py --mode features --features-path ../../../data/features/selected")
print("=" * 80)
