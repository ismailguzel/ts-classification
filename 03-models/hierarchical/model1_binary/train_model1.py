"""
Model 1: Binary Classification (Stationary vs Unstationary)
============================================================

Two training modes:
1. RAW MODE (default): Uses raw time series with sktime classifiers
   - TimeSeriesForestClassifier
   - ROCKET (Random Convolutional Kernel Transform)
   - Arsenal (ROCKET-based ensemble)
   
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

Refactored to integrate with utils module for FEATURES mode while preserving
RAW mode custom loaders for sktime compatibility.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import time
import argparse
import warnings
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
import gc
import json
from datetime import datetime

# Import utils for FEATURES mode and evaluation
from utils import (  # noqa: E402
    ModelEvaluator,
    load_features_and_labels,
    split_train_test,
    remove_series_id_leakage,
    BINARY_CLASS_NAMES,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
)

warnings.filterwarnings('ignore')

# Check optional dependencies
try:
    from xgboost import XGBClassifier
    has_xgboost = True
except ImportError:
    has_xgboost = False

try:
    from catboost import CatBoostClassifier
    has_catboost = True
except ImportError:
    has_catboost = False

# Parse arguments
parser = argparse.ArgumentParser(description='Train Model 1: Binary Classification')
parser.add_argument('--mode', type=str, default='raw', choices=['raw', 'features'],
                    help='Training mode: raw (sktime) or features (sklearn)')
parser.add_argument('--data-path', type=str, default='../../../data/raw/unified-5k',
                    help='Path to raw time series data')
parser.add_argument('--features-path', type=str, default='../../../data/features/unified-5k/selected',
                    help='Path to TSFresh features (for features mode)')
parser.add_argument('--test-size', type=float, default=0.2,
                    help='Test set size (default: 0.2)')
parser.add_argument('--random-state', type=int, default=42,
                    help='Random state for reproducibility')
parser.add_argument('--classifier', type=str, default='all',
                    choices=['all', 'tsf', 'rocket', 'arsenal'],
                    help='Specific classifier to train (default: all)')
parser.add_argument('--n-jobs', type=int, default=110,
                    help='Number of parallel jobs for model training (default: 110)')
parser.add_argument('--max-series', type=int, default=None,
                    help='Maximum total number of series to load (None = all)')
parser.add_argument('--balance', action='store_true',
                    help='Enable balanced sampling across classes (binary)')
parser.add_argument('--per-class', type=int, default=None,
                    help='Target number of series per class when --balance is set (overrides --max-series)')
parser.add_argument('--shuffle-files', action='store_true',
                    help='Shuffle file order before loading (recommended when using --balance)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for shuffling')
parser.add_argument('--save-dir', type=str, default='saved_models',
                    help='Directory to save model, metrics, and predictions')

args = parser.parse_args()

# Get n_jobs from arguments
N_JOBS = args.n_jobs

# Unified save directory with mode-specific subdirectory
base_save_dir = Path(args.save_dir).resolve()
if args.mode == 'raw':
    classifier_name = args.classifier.upper()
    SAVE_DIR = base_save_dir / f"model1_binary_{args.mode}_{classifier_name.lower()}"
else:  # features mode
    SAVE_DIR = base_save_dir / f"model1_binary_{args.mode}"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Fast parallel loading settings
BATCH_SIZE = 20      # Process 20 files per batch (increased from 10)
N_WORKERS = 16       # Parallel I/O threads (ThreadPoolExecutor)

print("="*80)
print("MODEL 1: BINARY CLASSIFICATION (Stationary vs Non-Stationary)")
print("="*80)
print(f"Mode: {args.mode.upper()}")
print(f"Data path: {args.data_path}")
if args.mode == 'features':
    print(f"Features path: {args.features_path}")
print(f"n_jobs: {N_JOBS}")
print("="*80)

# Use constant from utils for consistency
CLASS_NAMES = BINARY_CLASS_NAMES  # ['Stationary', 'Non-Stationary']


def safe_predict_proba(model, X):
    """Safely obtain predict_proba output when available."""
    if hasattr(model, 'predict_proba'):
        try:
            return model.predict_proba(X)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"      predict_proba unavailable: {exc}")
            return None
    return None

# ============================================================================
# Fast Parallel Data Loading Function (PyArrow + ThreadPoolExecutor)
# ============================================================================

def load_single_file(fp):
    """
    Load a single parquet file and extract all time series.
    Uses PyArrow for 5-10x faster reading than pandas.
    
    Returns:
        list of tuples: [(ts_data, label), ...]
    """
    try:
        # PyArrow table reading (C++ backend, very fast)
        table = pq.read_table(fp)
        df = table.to_pandas()
        
        # Validate columns
        if 'series_id' not in df.columns or 'data' not in df.columns:
            print(f"      Skipping {fp.name}: Missing required columns")
            return []
        
        if 'is_stationary' not in df.columns:
            print(f"      Skipping {fp.name}: Missing 'is_stationary' column")
            return []
        
        # Extract all series from this file
        series_list = []
        for series_id in df['series_id'].unique():
            series_data = df[df['series_id'] == series_id].sort_values('time')
            ts_data = series_data['data'].values
            label = 0 if series_data['is_stationary'].iloc[0] else 1
            series_list.append((series_id, ts_data, label))
        
        return series_list
        
    except Exception as e:
        print(f"      Error reading {fp.name}: {str(e)[:80]}")
        return []


def load_parquet_files_parallel(files, batch_size=BATCH_SIZE, n_workers=N_WORKERS,
                                max_series=None, balance=False, per_class=None, rng=None):
    """
    Load parquet files in parallel using PyArrow and ThreadPoolExecutor.
    
    Much faster than sequential pandas.read_parquet():
    - PyArrow: 5-10x faster file reading (C++ backend)
    - ThreadPoolExecutor: Parallel I/O (16 files simultaneously)
    - Result: ~10 minutes → ~2-3 minutes for 90K dataset
    
    Args:
        files: List of Path objects to parquet files
        batch_size: Number of files per progress update
        n_workers: Number of parallel I/O threads
    
    Returns:
        tuple: (series_list, labels, series_ids) - Ready for train/test split
    """
    print(f"\n Fast parallel loading with PyArrow (batch_size={batch_size}, workers={n_workers})")
    
    all_series = []
    all_labels = []
    all_ids = []
    class_counts = {0: 0, 1: 0}

    # Determine target limits
    target_total = None
    if balance and per_class is not None:
        target_total = per_class * 2
        print(f"  Target per-class: {per_class} (total {target_total})")
    elif max_series is not None:
        target_total = max_series
        print(f"  Target max series: {target_total}")
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(files) - 1) // batch_size + 1
            
            print(f"  Batch {batch_num}/{total_batches}: Processing {len(batch_files)} files...")
            
            # Parallel file reading
            results = executor.map(load_single_file, batch_files)
            
            # Collect results
            for series_list in results:
                for series_id, ts_data, label in series_list:
                    # Apply balanced or capped sampling if requested
                    if balance and per_class is not None:
                        # Keep until per-class cap reached
                        if class_counts[label] >= per_class:
                            continue
                        class_counts[label] += 1
                        all_series.append(ts_data)
                        all_labels.append(label)
                        all_ids.append(series_id)
                    elif target_total is not None:
                        if len(all_series) >= target_total:
                            break
                        all_series.append(ts_data)
                        all_labels.append(label)
                        all_ids.append(series_id)
                    else:
                        all_series.append(ts_data)
                        all_labels.append(label)
                        all_ids.append(series_id)

                # Early stop after finishing inner loop if limit reached
                if target_total is not None:
                    if balance and per_class is not None:
                        if class_counts[0] >= per_class and class_counts[1] >= per_class:
                            print(f"\n  Reached balanced target: {class_counts}")
                            return all_series, all_labels, all_ids
                    else:
                        if len(all_series) >= target_total:
                            print(f"\n  Reached max series target: {len(all_series)}")
                            return all_series[:target_total], all_labels[:target_total], all_ids[:target_total]
            
            # Show progress
            print(f"  Batch {batch_num}: Total series so far: {len(all_series):,}")
            
            # Memory cleanup every batch
            gc.collect()
    
    return all_series, all_labels, all_ids
print("="*80)

# ============================================================================
# 1. Load and Prepare Data
# ============================================================================

if args.mode == 'raw':
    # RAW MODE: Load time series from parquet files
    print("\n[1/6] Loading raw time series data...")
    
    raw_path = Path(args.data_path)
    if not raw_path.exists():
        print(f" Error: Data path not found: {raw_path}")
        print(f"    Please generate data first:")
        print(f"    cd ../../../01-data-generation && python generate_test.py")
        exit(1)
    
    # Search for parquet files recursively (they're in subdirectories by category)
    files = list(raw_path.rglob('*.parquet'))
    if not files:
        print(f" Error: No parquet files found in: {raw_path}")
        print(f"    Searched recursively in all subdirectories")
        print(f"    Expected structure: {raw_path}/stationary/, {raw_path}/deterministic_trend_*, etc.")
        exit(1)
    
    print(f"Found {len(files)} parquet files across categories")
    
    # Optional shuffle for balanced sampling
    if args.shuffle_files or args.balance:
        import random as _random
        _rng = _random.Random(args.seed)
        _rng.shuffle(files)
    
    # Load time series with fast parallel loading (PyArrow + ThreadPoolExecutor)
    series_list, labels, series_ids = load_parquet_files_parallel(
        files,
        batch_size=BATCH_SIZE,
        n_workers=N_WORKERS,
        max_series=(None if args.balance and args.per_class else args.max_series),
        balance=args.balance,
        per_class=args.per_class,
        rng=args.seed
    )
    
    print(f"\nLoaded {len(series_list):,} time series (parallel PyArrow loading)")
    print(f"  Labels: {len(labels):,}")
    print(f"  Memory-efficient: No full dataset loaded at once")
    if args.balance:
        import numpy as _np
        v, c = _np.unique(_np.array(labels), return_counts=True)
        print("  Class distribution after sampling:")
        for _v, _c in zip(v, c):
            print(f"    {'stationary' if _v==0 else 'non-stationary':>15}: {_c}")

    sample_ids = np.array(series_ids)

else:
    # FEATURES MODE: Load TSFresh features using utils
    print("\n[1/6] Loading TSFresh features...")
    
    features_path = Path(args.features_path)
    
    # Use utils function for consistent feature/label loading
    X_df, labels, labels_df = load_features_and_labels(
        features_path=features_path,
        target='binary'
    )
    
    # Extract sample_ids from index
    sample_ids = X_df.index.to_numpy()
    
    print(f"Loaded features: {X_df.shape}")
    print(f"Number of features: {X_df.shape[1]}")
    print(f"Number of samples: {X_df.shape[0]}")

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
    
    # Check for and handle NaN values
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"  Found {nan_count} NaN values in data, replacing with 0")
        X = np.nan_to_num(X, nan=0.0)
    
    # Check for inf values
    inf_count = np.isinf(X).sum()
    if inf_count > 0:
        print(f"  Found {inf_count} inf values in data, replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    X = X.reshape(X.shape[0], 1, X.shape[1])  # (n_samples, 1, n_timepoints)
    y = labels
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

else:
    # Features mode: Use TSFresh features directly
    print("\n[3/6] Preparing feature matrix...")
    
    # Import sklearn classifiers
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    
    # Optional dependencies already checked at top of file
    if not has_xgboost:
        print("  XGBoost not installed, skipping XGBoost model")
    if not has_catboost:
        print("  CatBoost not installed, skipping CatBoost model")
    
    # Don't scale yet - will scale after split to avoid data leakage
    X = X_features
    y = labels
    
    # Save feature names before converting to array (will be lost after scaling)
    feature_names = X_df.columns.tolist()
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Feature names: {len(feature_names)}")

# ============================================================================
# 3. Train/Test Split
# ============================================================================

print("\n[4/6] Splitting data...")
X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(
    X,
    y,
    sample_ids,
    test_size=args.test_size,
    random_state=args.random_state,
    stratify=y
)

train_ids = np.array(train_ids)
test_ids = np.array(test_ids)

# Scale features AFTER split to prevent data leakage
if args.mode == 'features':
    print("\n[4.5/6] Scaling features (fit on train only)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Fit only on training data
    X_test = scaler.transform(X_test)        # Transform test with train statistics
    print(f"Features scaled with StandardScaler (no data leakage)")

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
        print("\n Training TimeSeriesForestClassifier...")
        start_time = time.time()
        tsf = TimeSeriesForestClassifier(n_estimators=100, random_state=args.random_state, n_jobs=N_JOBS)
        tsf.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = tsf.predict(X_test)
        y_train_pred = tsf.predict(X_train)
        acc = accuracy_score(y_test, y_pred)
        train_acc = accuracy_score(y_train, y_train_pred)
        y_proba = safe_predict_proba(tsf, X_test)
        models['TimeSeriesForest'] = tsf
        results['TimeSeriesForest'] = {
            'accuracy': acc,
            'train_accuracy': train_acc,
            'train_time': train_time,
            'predictions': y_pred,
            'train_predictions': y_train_pred,
            'probabilities': y_proba
        }
        print(f"  ✓ Accuracy: {acc:.4f} ({100*acc:.2f}%)")
        print(f"  ✓ Train Accuracy: {train_acc:.4f} ({100*train_acc:.2f}%)")
        print(f"  ✓ Training time: {train_time:.2f}s")
        
        # Memory cleanup
        del tsf, y_pred, y_train_pred, y_proba
        import gc
        gc.collect()
        print("  Memory cleaned")
    
    # Model 2: ROCKET (SOTA)
    if args.classifier in ['all', 'rocket']:
        print("\n Training ROCKET Classifier...")
        try:
            start_time = time.time()
            rocket = RocketClassifier(num_kernels=1000, random_state=args.random_state, n_jobs=N_JOBS)
            rocket.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            y_pred = rocket.predict(X_test)
            y_train_pred = rocket.predict(X_train)
            acc = accuracy_score(y_test, y_pred)
            train_acc = accuracy_score(y_train, y_train_pred)
            y_proba = safe_predict_proba(rocket, X_test)
            models['ROCKET'] = rocket
            results['ROCKET'] = {
                'accuracy': acc,
                'train_accuracy': train_acc,
                'train_time': train_time,
                'predictions': y_pred,
                'train_predictions': y_train_pred,
                'probabilities': y_proba
            }
            print(f"  Accuracy: {acc:.4f} ({100*acc:.2f}%)")
            print(f"  Train Accuracy: {train_acc:.4f} ({100*train_acc:.2f}%)")
            print(f"  Training time: {train_time:.2f}s")
        except (AttributeError, ImportError) as e:
            print(f"    ROCKET not available: {str(e)[:100]}")
            print(f"    This may be due to NumPy 2.0 incompatibility. Consider downgrading to numpy<2.0")
    
    # Model 3: Arsenal (ROCKET ensemble)
    if args.classifier in ['all', 'arsenal']:
        print("\n Training Arsenal Classifier...")
        try:
            start_time = time.time()
            arsenal = Arsenal(num_kernels=1000, random_state=args.random_state, n_jobs=N_JOBS)
            arsenal.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            y_pred = arsenal.predict(X_test)
            y_train_pred = arsenal.predict(X_train)
            acc = accuracy_score(y_test, y_pred)
            train_acc = accuracy_score(y_train, y_train_pred)
            y_proba = safe_predict_proba(arsenal, X_test)
            models['Arsenal'] = arsenal
            results['Arsenal'] = {
                'accuracy': acc,
                'train_accuracy': train_acc,
                'train_time': train_time,
                'predictions': y_pred,
                'train_predictions': y_train_pred,
                'probabilities': y_proba
            }
            print(f"  Accuracy: {acc:.4f} ({100*acc:.2f}%)")
            print(f"  Train Accuracy: {train_acc:.4f} ({100*train_acc:.2f}%)")
            print(f"  Training time: {train_time:.2f}s")
        except (ImportError, AttributeError) as e:
            print(f"    Arsenal not available: {str(e)[:100]}")
            print(f"    This may be due to NumPy 2.0 incompatibility. Consider downgrading to numpy<2.0")

else:
    # FEATURES MODE: Train sklearn classifiers with ModelEvaluator
    
    # Model 1: Random Forest
    print("\n Training Random Forest...")
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=200, random_state=args.random_state, n_jobs=N_JOBS)
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    evaluator_rf = ModelEvaluator('RandomForest', rf, CLASS_NAMES, output_dir=SAVE_DIR)
    evaluator_rf.set_feature_names(feature_names)  # Set feature names for importance extraction
    evaluator_rf.set_train_time(train_time)
    results['RandomForest'] = evaluator_rf.evaluate(X_train, y_train, X_test, y_test, train_ids, test_ids)
    models['RandomForest'] = rf
    
    # Model 2: XGBoost (if available)
    if has_xgboost:
        print("\n Training XGBoost...")
        start_time = time.time()
        xgb = XGBClassifier(n_estimators=200, random_state=args.random_state, n_jobs=N_JOBS, 
                           eval_metric='logloss')
        xgb.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        evaluator_xgb = ModelEvaluator('XGBoost', xgb, CLASS_NAMES, output_dir=SAVE_DIR)
        evaluator_xgb.set_feature_names(feature_names)  # Set feature names for importance extraction
        evaluator_xgb.set_train_time(train_time)
        results['XGBoost'] = evaluator_xgb.evaluate(X_train, y_train, X_test, y_test, train_ids, test_ids)
        models['XGBoost'] = xgb
    
    # Model 3: CatBoost (if available)
    if has_catboost:
        print("\n Training CatBoost...")
        start_time = time.time()
        cat = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            thread_count=N_JOBS,
            random_seed=args.random_state,
            train_dir=str(SAVE_DIR / 'catboost_info'),  # Save CatBoost logs to save_dir
            verbose=0
        )
        cat.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        evaluator_cat = ModelEvaluator('CatBoost', cat, CLASS_NAMES, output_dir=SAVE_DIR)
        evaluator_cat.set_feature_names(feature_names)  # Set feature names for importance extraction
        evaluator_cat.set_train_time(train_time)
        results['CatBoost'] = evaluator_cat.evaluate(X_train, y_train, X_test, y_test, train_ids, test_ids)
        models['CatBoost'] = cat
    
    # Model 4: SVM (fast linear kernel for large feature sets)
    print("\n Training SVM (Linear)...")
    start_time = time.time()
    svm = SVC(kernel='linear', random_state=args.random_state, probability=True)
    svm.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    evaluator_svm = ModelEvaluator('SVM_Linear', svm, CLASS_NAMES, output_dir=SAVE_DIR)
    evaluator_svm.set_feature_names(feature_names)  # Set feature names (SVM doesn't have importance)
    evaluator_svm.set_train_time(train_time)
    results['SVM_Linear'] = evaluator_svm.evaluate(X_train, y_train, X_test, y_test, train_ids, test_ids)
    models['SVM_Linear'] = svm
# ============================================================================
# 5. Evaluate and Compare  
# ============================================================================

print("\n[6/6] Saving Results")
print("=" * 80)

# Save comprehensive metrics to JSON files
print(f"\nSaving results to: {SAVE_DIR}")

# Save individual model metrics
for model_name, result in results.items():
    metrics_path = SAVE_DIR / f"model1_{args.mode}_{model_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  {model_name} metrics: {metrics_path.name}")

# For RAW mode, manually save predictions and misclassified samples
# (FEATURES mode already handled by ModelEvaluator)
if args.mode == 'raw':
    print("\n Saving predictions for RAW mode models...")
    all_predictions = []
    
    for model_name, result in results.items():
        y_pred = result.get('predictions')
        y_proba = result.get('probabilities')
        
        if y_pred is not None:
            pred_data = {
                'id': test_ids,
                'true_label': y_test,
                'predicted_label': y_pred,
                'correct': (y_test == y_pred).astype(int),
                'model': model_name
            }
            
            # Add probability columns if available
            if y_proba is not None and len(y_proba.shape) == 2:
                for i, class_name in enumerate(CLASS_NAMES):
                    pred_data[f'prob_{class_name}'] = y_proba[:, i]
                pred_data['confidence'] = np.max(y_proba, axis=1)
            
            all_predictions.append(pd.DataFrame(pred_data))
    
    if all_predictions:
        # Save all predictions
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        predictions_path = SAVE_DIR / f"model1_{args.mode}_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        print(f"  All predictions: {predictions_path.name}")
        
        # Save misclassified samples
        misclassified_df = predictions_df[predictions_df['correct'] == 0].copy()
        if not misclassified_df.empty:
            misclassified_path = SAVE_DIR / f"model1_{args.mode}_misclassified.csv"
            misclassified_df.to_csv(misclassified_path, index=False)
            print(f"  Misclassified samples: {misclassified_path.name} ({len(misclassified_df)} errors)")

# Create summary comparison
summary = {
    'mode': args.mode,
    'timestamp_utc': datetime.utcnow().isoformat(timespec='seconds'),
    'models': {}
}

for model_name, result in results.items():
    # Handle both RAW mode (flat structure) and FEATURES mode (nested structure)
    if 'metrics' in result:
        # FEATURES mode: nested structure
        test_acc = result.get('metrics', {}).get('test', {}).get('accuracy', 0)
        train_acc = result.get('metrics', {}).get('train', {}).get('accuracy', 0)
    else:
        # RAW mode: flat structure
        test_acc = result.get('accuracy', 0)
        train_acc = result.get('train_accuracy', 0)
    
    summary['models'][model_name] = {
        'test_accuracy': float(test_acc),
        'train_accuracy': float(train_acc)
    }

summary_path = SAVE_DIR / f"model1_{args.mode}_summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Summary: {summary_path.name}")

# ============================================================================
# 6. Save Best Model
# ============================================================================

print("\n" + "=" * 80)
print("SAVING BEST MODEL")
print("=" * 80)

# Select best model based on test accuracy
def get_test_accuracy(result):
    """Get test accuracy from either RAW or FEATURES mode result structure"""
    if 'metrics' in result:
        return result.get('metrics', {}).get('test', {}).get('accuracy', 0)
    else:
        return result.get('accuracy', 0)

best_model_name = max(results, key=lambda x: get_test_accuracy(results[x]))
best_model = models[best_model_name]
best_accuracy = get_test_accuracy(results[best_model_name])

print(f"\n Best Model: {best_model_name}")
print(f"   Test Accuracy: {best_accuracy:.4f} ({100*best_accuracy:.2f}%)")

# Save model to SAVE_DIR
model_path = SAVE_DIR / 'model1_binary_classifier.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

print(f"\nModel saved to: {model_path}")

# Save metadata
metadata = {
    'model_name': best_model_name,
    'mode': args.mode,
    'test_accuracy': best_accuracy,
    'train_accuracy': results[best_model_name].get('metrics', {}).get('train', {}).get('accuracy', 0),
    'n_train': len(X_train),
    'n_test': len(X_test),
    'classes': CLASS_NAMES,
    'save_dir': str(SAVE_DIR),
}

# Add mode-specific metadata
if args.mode == 'raw':
    metadata['fixed_length'] = fixed_length
    metadata['feature_shape'] = X_train.shape[1:]  # Use X_train shape after split
elif args.mode == 'features':
    metadata['n_features'] = X_train.shape[1]  # Use X_train shape after split
    # Note: scaler object saved separately if needed for inference

# Save scaler separately for features mode (needed for inference)
if args.mode == 'features':
    scaler_path = SAVE_DIR / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    metadata['scaler_path'] = str(scaler_path)
    print(f"Scaler saved to: {scaler_path}")

metadata_path = SAVE_DIR / 'model1_metadata.pkl'
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)

print(f"Metadata saved to: {metadata_path}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print(" MODEL 1 TRAINING COMPLETE!")
print("=" * 80)
print(f"\nMode: {args.mode.upper()}")
print(f"Best Model: {best_model_name}")
print(f"Accuracy: {100*best_accuracy:.2f}%")

print(f"\n Output Structure:")
print(f"   {SAVE_DIR}/")
print(f"   ├── model1_binary_classifier.pkl")
print(f"   ├── model1_metadata.pkl")
print(f"   ├── predictions_RandomForest.csv")
print(f"   ├── predictions_XGBoost.csv")
print(f"   ├── predictions_CatBoost.csv")
print(f"   ├── predictions_SVM_Linear.csv")
print(f"   ├── model1_{args.mode}_RandomForest_metrics.json")
print(f"   ├── model1_{args.mode}_XGBoost_metrics.json")
print(f"   ├── model1_{args.mode}_CatBoost_metrics.json")
print(f"   ├── model1_{args.mode}_SVM_Linear_metrics.json")
print(f"   ├── model1_{args.mode}_summary.json")
if has_catboost:
    print(f"   └── catboost_info/")

print("\nNext steps:")
print("  1. Review classification report above")
print("  2. Test model: python test_model1.py")
print("  3. If accuracy is good (>90%), proceed to Model 2 (primary categories)")
if args.mode == 'raw':
    print("\nTip: Try features mode for potentially better performance:")
    print("  python train_model1.py --mode features --features-path ../../../data/features/selected")
print("=" * 80)

