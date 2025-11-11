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
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
import gc
warnings.filterwarnings('ignore')
import time
import argparse
import warnings
warnings.filterwarnings('ignore')
import json
from datetime import datetime


def ensure_series_column(df: pd.DataFrame, context: str) -> pd.DataFrame:
    if 'series_id' in df.columns:
        return df
    if 'id' in df.columns:
        return df.rename(columns={'id': 'series_id'})
    raise ValueError(f"{context} requires a 'series_id' column; columns: {list(df.columns)[:5]}")


def set_series_index(df: pd.DataFrame, context: str) -> pd.DataFrame:
    df = ensure_series_column(df, context)
    return df.set_index('series_id')
# Parse arguments
parser = argparse.ArgumentParser(description='Train Model 2: Non-Stationary 5-Class Classification')
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
parser.add_argument('--output-dir', type=str, default=None,
                    help='Directory to store structured metrics and predictions')

args = parser.parse_args()

# Get n_jobs from arguments
N_JOBS = args.n_jobs

OUTPUT_DIR = Path(args.output_dir).resolve() if args.output_dir else None
if OUTPUT_DIR:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Global n_jobs setting for all models

# Fast parallel loading settings
BATCH_SIZE = 20      # Process 20 files per batch
N_WORKERS = 16       # Parallel I/O threads (ThreadPoolExecutor)

print("="*80)
print("MODEL 2: NON-STATIONARY 5-CLASS CLASSIFICATION")
print("="*80)
print(f"Mode: {args.mode.upper()}")
print(f"Data path: {args.data_path}")
if args.mode == 'features':
    print(f"Features path: {args.features_path}")
print(f"n_jobs: {N_JOBS}")
print("="*80)


def safe_predict_proba(model, X):
    if hasattr(model, 'predict_proba'):
        try:
            return model.predict_proba(X)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"    ⚠️  predict_proba unavailable: {exc}")
            return None
    return None

# ============================================================================
# Fast Parallel Data Loading Function (PyArrow + ThreadPoolExecutor)
# ============================================================================

def load_single_file(fp):
    """
    Load a single parquet file and extract all NON-STATIONARY time series.
    Uses PyArrow for 5-10x faster reading than pandas.
    
    Returns:
    list of tuples: [(series_id, ts_data, label), ...]
    """
    try:
        # PyArrow table reading (C++ backend, very fast)
        table = pq.read_table(fp)
        df = table.to_pandas()
        
        # Validate columns
        if 'series_id' not in df.columns or 'data' not in df.columns:
            return []
        
        if 'is_stationary' not in df.columns or 'primary_category' not in df.columns:
            return []
        
        # Filter for non-stationary only
        df = df[df['is_stationary'] == False].copy()
        
        if len(df) == 0:
            return []
        
        # Extract all series from this file
        series_list = []
        for series_id in df['series_id'].unique():
            series_data = df[df['series_id'] == series_id].sort_values('time')
            ts_data = series_data['data'].values
            primary_cat = series_data['primary_category'].iloc[0]
            
            # Map category to numeric label
            if primary_cat in CATEGORY_MAPPING:
                label = CATEGORY_MAPPING[primary_cat]
                series_list.append((series_id, ts_data, label))
        
        return series_list
        
    except Exception as e:
        print(f"    ⚠️  Error reading {fp.name}: {str(e)[:80]}")
        return []


def load_parquet_files_parallel(files, batch_size=BATCH_SIZE, n_workers=N_WORKERS):
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
    print(f"\n⚡ Fast parallel loading with PyArrow (batch_size={batch_size}, workers={n_workers})")
    
    all_series = []
    all_labels = []
    all_ids = []
    
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
                    all_series.append(ts_data)
                    all_labels.append(label)
                    all_ids.append(series_id)
            
            # Show progress
            print(f"  ✓ Batch {batch_num}: Total series so far: {len(all_series):,}")
            
            # Memory cleanup every batch
            gc.collect()
    
    return all_series, all_labels, all_ids

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
    
    # Load time series with fast parallel loading (PyArrow + ThreadPoolExecutor)
    series_list, labels, series_ids = load_parquet_files_parallel(files)
    
    if len(series_list) == 0:
        print("❌ Error: No non-stationary series found!")
        exit(1)
    
    print(f"\n✓ Loaded {len(series_list):,} non-stationary time series (parallel PyArrow loading)")
    print(f"  Labels: {len(labels):,}")
    print(f"  Memory-efficient: No full dataset loaded at once")
    
    print(f"\n✓ Prepared {len(series_list):,} time series without loading full dataset")
    sample_ids = np.array(series_ids)
    
    # Display label distribution
    print("\n[2/6] Analyzing label distribution...")
    unique_labels, counts = np.unique(labels, return_counts=True)

else:
    # FEATURES MODE: Load TSFresh features
    print("\n[1/6] Loading TSFresh features...")
    
    features_path = Path(args.features_path)

    candidate_pairs = [
        (features_path / 'features.parquet', features_path / 'labels.parquet'),
        (features_path / 'features_primary_mutual_info.parquet', features_path / 'labels_primary.parquet'),
        (features_path / 'primary/features.parquet', features_path / 'primary/labels.parquet')
    ]

    features_file = None
    labels_file = None
    for feat_file, lab_file in candidate_pairs:
        if feat_file.exists() and lab_file.exists():
            features_file, labels_file = feat_file, lab_file
            break

    if features_file is None or labels_file is None:
        print(f"❌ Error: Features or labels file not found under {features_path}")
        print("    Expected combinations: features.parquet+labels.parquet,"
              " primary/ directory, or legacy mutual-info selection.")
        exit(1)

    X_df = set_series_index(pd.read_parquet(features_file), "Features table")
    labels_df = ensure_series_column(pd.read_parquet(labels_file), "Labels table").set_index('series_id')
    labels_df = labels_df.reindex(X_df.index)

    if labels_df.isnull().any().any():
        missing_ids = labels_df.index[labels_df.isnull().any(axis=1)].tolist()
        print(f"❌ Error: Missing labels for series IDs: {missing_ids[:5]}")
        exit(1)

    if 'is_stationary' not in labels_df.columns:
        print("❌ Error: 'is_stationary' column not found in labels table")
        exit(1)

    nonstat_mask = labels_df['is_stationary'] == False
    X_df = X_df[nonstat_mask]
    labels_df = labels_df[nonstat_mask]

    print(f"✓ Filtered to {len(X_df):,} non-stationary series")

    if X_df.empty:
        print("❌ Error: No non-stationary series found!")
        exit(1)

    if 'primary_category' not in labels_df.columns:
        print("❌ Error: 'primary_category' column missing in labels table")
        exit(1)

    labels = labels_df['primary_category'].map(CATEGORY_MAPPING).values
    
    print(f"✓ Loaded features: {X_df.shape}")
    print(f"✓ Number of features: {X_df.shape[1]}")
    print(f"✓ Number of samples: {X_df.shape[0]}")
    
    # Store for later use
    X_features = X_df.values
    series_list = None  # Not used in features mode
    sample_ids = X_df.index.to_numpy()

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
    
    # Optional: CatBoost
    try:
        from catboost import CatBoostClassifier
        has_catboost = True
    except ImportError:
        has_catboost = False
        print("⚠️  CatBoost not installed, skipping CatBoost model")
    
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
    
    # Model 2: ROCKET (2000 kernels for 5-class)
    if args.classifier in ['all', 'rocket']:
        print("\n🚀 Training ROCKET Classifier...")
        try:
            start_time = time.time()
            rocket = RocketClassifier(num_kernels=2000, random_state=args.random_state, n_jobs=N_JOBS)
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
            print(f"  ✓ Accuracy: {acc:.4f} ({100*acc:.2f}%)")
            print(f"  ✓ Train Accuracy: {train_acc:.4f} ({100*train_acc:.2f}%)")
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
            print(f"  ✓ Accuracy: {acc:.4f} ({100*acc:.2f}%)")
            print(f"  ✓ Train Accuracy: {train_acc:.4f} ({100*train_acc:.2f}%)")
            print(f"  ✓ Training time: {train_time:.2f}s")
        except (ImportError, AttributeError) as e:
            print(f"  ⚠️  Arsenal not available: {str(e)[:100]}")
            print(f"  ⚠️  This may be due to NumPy 2.0 incompatibility. Consider downgrading to numpy<2.0")

else:
    # FEATURES MODE: Train sklearn classifiers
    
    # Model 1: Random Forest
    print("\n🌲 Training Random Forest...")
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=200, random_state=args.random_state, n_jobs=N_JOBS)
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = rf.predict(X_test)
    y_train_pred = rf.predict(X_train)
    acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    y_proba = safe_predict_proba(rf, X_test)
    models['RandomForest'] = rf
    results['RandomForest'] = {
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
    
    # Model 2: XGBoost (if available)
    if has_xgboost:
        print("\n🚀 Training XGBoost...")
        start_time = time.time()
        xgb = XGBClassifier(n_estimators=200, random_state=args.random_state, n_jobs=N_JOBS, 
                           eval_metric='mlogloss')
        xgb.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = xgb.predict(X_test)
        y_train_pred = xgb.predict(X_train)
        acc = accuracy_score(y_test, y_pred)
        train_acc = accuracy_score(y_train, y_train_pred)
        y_proba = safe_predict_proba(xgb, X_test)
        models['XGBoost'] = xgb
        results['XGBoost'] = {
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
    
    # Model 3: CatBoost (if available)
    if has_catboost:
        print("\n🐱 Training CatBoost...")
        start_time = time.time()
        cat = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            thread_count=N_JOBS,
            random_seed=args.random_state,
            verbose=0
        )
        cat.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = cat.predict(X_test)
        y_train_pred = cat.predict(X_train)
        acc = accuracy_score(y_test, y_pred)
        train_acc = accuracy_score(y_train, y_train_pred)
        y_proba = safe_predict_proba(cat, X_test)
        models['CatBoost'] = cat
        results['CatBoost'] = {
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
    
    # Model 4: SVM (RBF kernel for multi-class)
    print("\n⚡ Training SVM (RBF)...")
    start_time = time.time()
    svm = SVC(kernel='rbf', random_state=args.random_state, probability=True)
    svm.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = svm.predict(X_test)
    y_train_pred = svm.predict(X_train)
    acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    y_proba = safe_predict_proba(svm, X_test)
    models['SVM_RBF'] = svm
    results['SVM_RBF'] = {
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

# ============================================================================
# 5. Evaluate and Compare
# ============================================================================

print("\n[6/6] Evaluation Results")
print("=" * 80)

metrics_payload = {
    'mode': args.mode,
    'timestamp_utc': datetime.utcnow().isoformat(timespec='seconds'),
    'n_train': int(len(X_train)),
    'n_test': int(len(X_test)),
    'train_sample_ids': train_ids.tolist(),
    'test_sample_ids': test_ids.tolist(),
    'class_names': CLASS_NAMES,
    'models': {}
}

prediction_frames = []

for model_name, result in results.items():
    print(f"\n📊 {model_name}")
    print("-" * 80)
    acc = float(result['accuracy'])
    train_acc_value = result.get('train_accuracy', np.nan)
    train_acc = float(train_acc_value) if train_acc_value is not None else np.nan
    train_time = float(result['train_time'])
    print(f"Accuracy      : {acc:.4f} ({100*acc:.2f}%)")
    if not np.isnan(train_acc):
        print(f"Train Accuracy: {train_acc:.4f} ({100*train_acc:.2f}%)")
    print(f"Training Time : {train_time:.2f}s")

    y_pred = np.array(result['predictions'])
    y_train_pred = np.array(result.get('train_predictions', []))
    y_proba = result.get('probabilities')

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
    print("\nClassification Report:")
    print(report_text)

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    header = " " * 20 + " ".join([f"{name[:8]:>8s}" for name in CLASS_NAMES])
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row_vals = " ".join([f"{cm[i, j]:8d}" for j in range(len(CLASS_NAMES))])
        print(f"{name[:20]:20s} {row_vals}")

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    if y_train_pred.size:
        train_report_dict = classification_report(
            y_train,
            y_train_pred,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0
        )
        train_cm = confusion_matrix(y_train, y_train_pred).tolist()
    else:
        train_report_dict = {}
        train_cm = []

    y_pred_flat = np.asarray(y_pred).ravel()
    y_test_flat = np.asarray(y_test).ravel()
    test_ids_arr = np.asarray(test_ids).ravel()

    misclassified_mask = (y_pred_flat != y_test_flat)
    misclassified_count = int(np.sum(misclassified_mask))
    if misclassified_count > 0:
        preview_ids = test_ids_arr[misclassified_mask][:5]
        print(f"  ⚠️  Misclassified samples: {misclassified_count} (examples: {preview_ids})")
    else:
        print("  ✓ No misclassifications on test fold")

    model_metrics = {
        'train_time_sec': train_time,
        'test_accuracy': acc,
        'train_accuracy': None if np.isnan(train_acc) else train_acc,
        'classification_report': report_dict,
        'confusion_matrix': cm.tolist(),
        'train_classification_report': train_report_dict,
        'train_confusion_matrix': train_cm,
        'misclassified_count': misclassified_count,
        'probability_available': y_proba is not None
    }

    if y_proba is not None:
        model_metrics['probability_shape'] = list(np.shape(y_proba))

    metrics_payload['models'][model_name] = model_metrics

    pred_df = pd.DataFrame({
        'sample_id': test_ids_arr,
        'true_label': y_test_flat,
        'predicted_label': y_pred_flat,
        'model_name': model_name
    })

    if y_proba is not None:
        proba_array = np.asarray(y_proba)
        for idx, cls_name in enumerate(CLASS_NAMES):
            col_name = f"prob_{cls_name.lower().replace(' ', '_')}"
            pred_df[col_name] = proba_array[:, idx]

    prediction_frames.append(pred_df)

if prediction_frames:
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    misclassified_df = predictions_df[predictions_df['true_label'] != predictions_df['predicted_label']].copy()
else:
    predictions_df = pd.DataFrame()
    misclassified_df = pd.DataFrame()

artifacts = {}

if OUTPUT_DIR:
    metrics_path = OUTPUT_DIR / f"model2_{args.mode}_metrics.json"
    artifacts['metrics_json'] = str(metrics_path)

    if not predictions_df.empty:
        predictions_path = OUTPUT_DIR / f"model2_{args.mode}_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        artifacts['predictions_csv'] = str(predictions_path)

        if not misclassified_df.empty:
            misclassified_path = OUTPUT_DIR / f"model2_{args.mode}_misclassified.csv"
            misclassified_df.to_csv(misclassified_path, index=False)
            artifacts['misclassified_csv'] = str(misclassified_path)

    metrics_payload['artifacts'] = artifacts
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=2)
else:
    metrics_payload['artifacts'] = artifacts

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

if OUTPUT_DIR:
    metadata['metrics_output_dir'] = str(OUTPUT_DIR)

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
