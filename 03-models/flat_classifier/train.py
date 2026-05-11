"""
Flat Time Series Classification
=================================

Trains sklearn classifiers on extracted features.
Supports any number of classes — works with full (39), test (39), and topo-test (10).

Classifiers: Random Forest, XGBoost, CatBoost, SVM (RBF)

Feature modes:
    TSFresh only   : --features-path <tsfresh_selected>
    Topology only  : --features-path <topo_selected>
    Hybrid (merged): --features-path <tsfresh_selected> --extra-features-path <topo_selected>

Usage:
    python train.py --features-path ../../data/features/topo-test/selected --save-dir output
    python train.py --features-path ../../data/features/topo-test/topological_selected --save-dir output_topo
    python train.py --features-path ../../data/features/topo-test/selected \\
                    --extra-features-path ../../data/features/topo-test/topological_selected \\
                    --save-dir output_hybrid
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pickle
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import argparse

from utils import (  # noqa: E402
    ModelEvaluator,
    load_features_and_labels,
    split_train_test,
    remove_series_id_leakage,
    PRIMARY_CATEGORY_MAPPING,
    PRIMARY_CLASS_NAMES,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
)

warnings.filterwarnings('ignore')

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

# ── Arguments ────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='Train flat time series classifier')
parser.add_argument('--features-path', type=str,
                    default='../../data/features/dataset/selected',
                    help='Primary feature directory (TSFresh or topology selected)')
parser.add_argument('--extra-features-path', type=str, default=None,
                    help='Optional second feature directory; when provided, both sets '
                         'are merged on series_id (hybrid mode)')
parser.add_argument('--test-size', type=float, default=DEFAULT_TEST_SIZE)
parser.add_argument('--random-state', type=int, default=DEFAULT_RANDOM_STATE)
parser.add_argument('--classifier', type=str, default='all',
                    choices=['all', 'rf', 'xgboost', 'catboost', 'svm'])
parser.add_argument('--n-jobs', type=int, default=-1)
parser.add_argument('--save-dir', type=str, default='output',
                    help='Directory to save models, metrics, and predictions')

args = parser.parse_args()

N_JOBS = args.n_jobs
SAVE_DIR = Path(args.save_dir).resolve()
SAVE_DIR.mkdir(parents=True, exist_ok=True)

HYBRID = args.extra_features_path is not None
FEATURE_TYPE = 'hybrid' if HYBRID else (
    'topo' if 'topological' in str(args.features_path).lower() else 'tsfresh'
)

print("=" * 80)
print("FLAT CLASSIFICATION")
print("=" * 80)
print(f"Feature type  : {FEATURE_TYPE}")
print(f"Features path : {args.features_path}")
if HYBRID:
    print(f"Extra features: {args.extra_features_path}")
print(f"Classifier    : {args.classifier}")
print(f"n_jobs        : {N_JOBS}")
print(f"Save dir      : {SAVE_DIR}")
print("=" * 80)

# ── 1. Load Features ─────────────────────────────────────────────────────────

print(f"\n[1/5] Loading features ({FEATURE_TYPE})...")

X_df, labels, labels_df = load_features_and_labels(
    features_path=Path(args.features_path),
    target='primary',
)

if HYBRID:
    print(f"  Primary : {X_df.shape}")
    X_extra, _, _ = load_features_and_labels(
        features_path=Path(args.extra_features_path),
        target='primary',
    )
    print(f"  Extra   : {X_extra.shape}")
    # Merge on series_id index — keep only series present in both
    X_df = X_df.join(X_extra, how='inner', lsuffix='', rsuffix='_extra')
    # Re-align labels to merged series (encode strings → ints via mapping)
    labels_df = labels_df[labels_df.index.isin(X_df.index)]
    labels = labels_df['primary_category'].map(PRIMARY_CATEGORY_MAPPING).values
    print(f"  Merged  : {X_df.shape}  ({len(X_df)} series)")

sample_ids = X_df.index.to_numpy()
feature_names = X_df.columns.tolist()
X = X_df.values
y = np.array(labels)

present_labels = sorted(np.unique(y))
N_CLASSES = len(present_labels)
# Build class name list for only the classes present in the data
CLASS_NAMES = [PRIMARY_CLASS_NAMES[i] for i in present_labels]

print(f"\nFeature matrix : {X.shape}")
print(f"Classes        : {N_CLASSES}")

# ── 2. Train / Test Split ────────────────────────────────────────────────────

print("\n[2/5] Splitting data (stratified)...")

X_train, X_test, y_train, y_test, train_ids, test_ids = split_train_test(
    X, y,
    sample_ids=sample_ids,
    test_size=args.test_size,
    random_state=args.random_state,
    stratify=True,
)

print(f"Train : {len(X_train):,}   Test : {len(X_test):,}")

# Scale after split — no leakage
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 3. Train Models ──────────────────────────────────────────────────────────

print("\n[3/5] Training models...")

models  = {}
results = {}


def _train(name, clf):
    print(f"\n  ▸ {name}")
    t0 = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0
    ev = ModelEvaluator(name, clf, CLASS_NAMES, output_dir=SAVE_DIR)
    ev.set_feature_names(feature_names)
    ev.set_train_time(elapsed)
    res = ev.evaluate(X_train, y_train, X_test, y_test, train_ids, test_ids)
    models[name]  = clf
    results[name] = res
    acc = res.get('metrics', {}).get('test', {}).get('accuracy', 0)
    print(f"    accuracy={acc:.4f}  time={elapsed:.1f}s")


if args.classifier in ('all', 'rf'):
    _train('RandomForest',
           RandomForestClassifier(n_estimators=200, random_state=args.random_state,
                                  n_jobs=N_JOBS))

if args.classifier in ('all', 'xgboost') and has_xgboost:
    _train('XGBoost',
           XGBClassifier(n_estimators=200, random_state=args.random_state,
                         n_jobs=N_JOBS, eval_metric='mlogloss'))
elif args.classifier in ('all', 'xgboost'):
    print("  XGBoost not installed — skipping")

if args.classifier in ('all', 'catboost') and has_catboost:
    _train('CatBoost',
           CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1,
                              thread_count=N_JOBS, random_seed=args.random_state,
                              train_dir=str(SAVE_DIR / 'catboost_info'),
                              verbose=0))
elif args.classifier in ('all', 'catboost'):
    print("  CatBoost not installed — skipping")

if args.classifier in ('all', 'svm'):
    _train('SVM_RBF',
           SVC(kernel='rbf', random_state=args.random_state, probability=True))

# ── 4. Save Metrics ──────────────────────────────────────────────────────────

print("\n[4/5] Saving metrics...")

for name, res in results.items():
    p = SAVE_DIR / f"metrics_{name}.json"
    with open(p, 'w') as f:
        json.dump(res, f, indent=2, default=str)

summary = {
    'timestamp_utc': datetime.utcnow().isoformat(timespec='seconds'),
    'feature_type': FEATURE_TYPE,
    'n_features': X_train.shape[1],
    'n_classes': N_CLASSES,
    'n_train': len(X_train),
    'n_test': len(X_test),
    'models': {
        name: {
            'test_accuracy':  res.get('metrics', {}).get('test', {}).get('accuracy', 0),
            'train_accuracy': res.get('metrics', {}).get('train', {}).get('accuracy', 0),
        }
        for name, res in results.items()
    },
}
with open(SAVE_DIR / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# ── 5. Save Best Model ───────────────────────────────────────────────────────

print("\n[5/5] Saving best model...")

best_name = max(results, key=lambda n: results[n].get('metrics', {}).get('test', {}).get('accuracy', 0))
best_acc  = results[best_name].get('metrics', {}).get('test', {}).get('accuracy', 0)

model_path = SAVE_DIR / 'classifier.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(models[best_name], f)

scaler_path = SAVE_DIR / 'scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

metadata = {
    'model_name':    best_name,
    'feature_type':  FEATURE_TYPE,
    'test_accuracy': best_acc,
    'n_classes':     N_CLASSES,
    'class_indices': present_labels,           # actual label ints used in training
    'class_names':   CLASS_NAMES,              # only present classes, not all 39
    'n_features':    X_train.shape[1],
    'feature_names': feature_names,
    'n_train':       len(X_train),
    'n_test':        len(X_test),
    'save_dir':      str(SAVE_DIR),
    'scaler_path':   str(scaler_path),
}
with open(SAVE_DIR / 'metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

# ── Summary ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Best model : {best_name}  ({100*best_acc:.2f}%)")
print(f"Output dir : {SAVE_DIR}")
print("=" * 80)
