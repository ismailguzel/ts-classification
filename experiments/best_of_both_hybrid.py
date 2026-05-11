"""
Best-of-Both Hybrid Experiment
================================
Tests whether selecting top-K TSFresh + top-M topology features outperforms
the current hybrid (all 100 TSFresh + all topo features).

Feature importances are derived from models already trained on individual
TSFresh and topology pipelines (output/ and output_topo/).

Usage:
    python best_of_both_hybrid.py
    python best_of_both_hybrid.py --mode topo-test
    python best_of_both_hybrid.py --mode topo-test --classifier xgboost
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent

parser = argparse.ArgumentParser(description='Best-of-both hybrid feature selection experiment')
parser.add_argument('--mode', default='topo-test', choices=['full', 'test', 'topo-test'])
parser.add_argument('--classifier', default='all', choices=['all', 'rf', 'xgboost', 'catboost'])
parser.add_argument('--random-state', type=int, default=42)
parser.add_argument('--test-size', type=float, default=0.2)
args = parser.parse_args()

TSFRESH_FEAT_DIR  = ROOT / 'data/features' / args.mode / 'selected'
TOPO_FEAT_DIR     = ROOT / 'data/features' / args.mode / 'topological_selected'
TSFRESH_MODEL_DIR = ROOT / '03-models/flat_classifier/output'
TOPO_MODEL_DIR    = ROOT / '03-models/flat_classifier/output_topo'

sys.path.insert(0, str(ROOT / '03-models'))
from utils import load_features_and_labels, split_train_test, PRIMARY_CATEGORY_MAPPING

# ── Load features ─────────────────────────────────────────────────────────────

print("=" * 70)
print(f"BEST-OF-BOTH HYBRID  |  mode={args.mode}  |  clf={args.classifier}")
print("=" * 70)

print("\n[1/4] Loading features...")

X_ts,   labels,    labels_df = load_features_and_labels(TSFRESH_FEAT_DIR, target='primary')
X_topo, labels_tp, _         = load_features_and_labels(TOPO_FEAT_DIR,    target='primary')

print(f"  TSFresh : {X_ts.shape}")
print(f"  Topo    : {X_topo.shape}")

# Align on common series_ids
common_ids = X_ts.index.intersection(X_topo.index)
X_ts   = X_ts.loc[common_ids]
X_topo = X_topo.loc[common_ids]
labels_df = labels_df.loc[common_ids]
y = labels_df['primary_category'].map(PRIMARY_CATEGORY_MAPPING).values
print(f"  Common  : {len(common_ids)} series")

# ── Load feature importances ──────────────────────────────────────────────────

print("\n[2/4] Computing MI scores on train split...")

# Split first so MI is fitted on train only — no leakage
sample_ids_tmp = np.array(X_ts.index)
y_tmp = labels_df['primary_category'].map(PRIMARY_CATEGORY_MAPPING).values

sss_tmp = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
train_pos_tmp, _ = next(sss_tmp.split(np.zeros(len(y_tmp)), y_tmp))

X_ts_train   = X_ts.values[train_pos_tmp]
X_topo_train = X_topo.values[train_pos_tmp]
y_train_tmp  = y_tmp[train_pos_tmp]

mi_ts   = mutual_info_classif(X_ts_train,   y_train_tmp, random_state=args.random_state)
mi_topo = mutual_info_classif(X_topo_train, y_train_tmp, random_state=args.random_state)

imp_ts   = pd.Series(mi_ts,   index=X_ts.columns).sort_values(ascending=False)
imp_topo = pd.Series(mi_topo, index=X_topo.columns).sort_values(ascending=False)

print(f"  TSFresh MI scores : {len(imp_ts)} features  (top: {imp_ts.index[0]}={imp_ts.iloc[0]:.4f})")
print(f"  Topo    MI scores : {len(imp_topo)} features  (top: {imp_topo.index[0]}={imp_topo.iloc[0]:.4f})")

# ── Split (same split for all experiments) ───────────────────────────────────

print("\n[3/4] Splitting data (stratified, fixed seed)...")

sample_ids = np.array(common_ids)

# Stratified split — returns positional indices into sample_ids
sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
train_pos, test_pos = next(sss.split(np.zeros(len(y)), y))

y_train = y[train_pos]
y_test  = y[test_pos]

print(f"  Train: {len(train_pos)}  Test: {len(test_pos)}")

# ── Classifier factory ────────────────────────────────────────────────────────

def make_classifiers() -> dict:
    clfs = {}
    if args.classifier in ('all', 'rf'):
        clfs['RandomForest'] = RandomForestClassifier(
            n_estimators=200, random_state=args.random_state, n_jobs=-1)
    if args.classifier in ('all', 'xgboost') and HAS_XGBOOST:
        clfs['XGBoost'] = XGBClassifier(
            n_estimators=200, random_state=args.random_state,
            n_jobs=-1, eval_metric='mlogloss', verbosity=0)
    if args.classifier in ('all', 'catboost') and HAS_CATBOOST:
        clfs['CatBoost'] = CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.1,
            random_seed=args.random_state, verbose=0)
    return clfs

def evaluate(X):
    """Train/test split using positional indices, return per-classifier accuracies."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[train_pos])
    Xte = scaler.transform(X[test_pos])
    accs = {}
    for name, clf in make_classifiers().items():
        clf.fit(Xtr, y_train)
        accs[name] = accuracy_score(y_test, clf.predict(Xte))
    mean_acc = np.mean(list(accs.values()))
    return accs, mean_acc

# ── Experiment ────────────────────────────────────────────────────────────────

print("\n[4/4] Running experiments...\n")

X_ts_arr   = X_ts.values
X_topo_arr = X_topo.values

results = []

# Baseline A — all TSFresh
acc_ts, mean_ts = evaluate(X_ts_arr)
results.append({'label': 'TSFresh (all 100)', 'n_ts': len(imp_ts), 'n_topo': 0,
                'mean_acc': mean_ts, **{f'acc_{k}': v for k, v in acc_ts.items()}})

# Baseline B — all Topo
acc_tp, mean_tp = evaluate(X_topo_arr)
results.append({'label': f'Topo (all {len(imp_topo)})', 'n_ts': 0, 'n_topo': len(imp_topo),
                'mean_acc': mean_tp, **{f'acc_{k}': v for k, v in acc_tp.items()}})

# Baseline C — current hybrid (all TSFresh + all Topo)
acc_hy, mean_hy = evaluate(np.hstack([X_ts_arr, X_topo_arr]))
results.append({'label': f'Hybrid (100 + {len(imp_topo)})', 'n_ts': len(imp_ts), 'n_topo': len(imp_topo),
                'mean_acc': mean_hy, **{f'acc_{k}': v for k, v in acc_hy.items()}})

# Sweep — top-K TSFresh × top-M Topo
n_topo_total   = len(imp_topo)
top_ts_options = [10, 20, 30, 50, 70]

for k_ts in top_ts_options:
    ts_cols  = imp_ts.head(k_ts).index.tolist()
    X_ts_top = X_ts[ts_cols].values

    best_m, best_mean, best_accs = None, -1, {}
    for m_tp in range(1, n_topo_total + 1):
        tp_cols    = imp_topo.head(m_tp).index.tolist()
        X_topo_top = X_topo[tp_cols].values
        accs, mean_acc = evaluate(np.hstack([X_ts_top, X_topo_top]))
        if mean_acc > best_mean:
            best_mean, best_m, best_accs = mean_acc, m_tp, accs

    label = f'Top-{k_ts} TSFresh + Top-{best_m} Topo'
    results.append({'label': label, 'n_ts': k_ts, 'n_topo': best_m,
                    'mean_acc': best_mean,
                    **{f'acc_{k}': v for k, v in best_accs.items()}})

# ── Print results ─────────────────────────────────────────────────────────────

clf_names = [k for k in make_classifiers()]

header = f"{'Configuration':40s}  {'Feats':>6}  {'Mean':>7}"
for c in clf_names:
    header += f"  {c[:10]:>10}"
print(header)
print('-' * (len(header) + 5))

for r in sorted(results, key=lambda x: x['mean_acc'], reverse=True):
    n_total = r['n_ts'] + r['n_topo']
    row = f"{r['label']:40s}  {n_total:>6}  {r['mean_acc']:>7.1%}"
    for c in clf_names:
        row += f"  {r.get(f'acc_{c}', 0):>10.1%}"
    # mark if better than full hybrid
    if r['mean_acc'] > mean_hy + 0.001:
        row += '  ← better'
    print(row)

print()
print(f"Baseline hybrid mean acc : {mean_hy:.3f}")
best_sweep = max((r for r in results if r['n_ts'] not in (len(imp_ts), 0) or r['n_topo'] not in (n_topo_total, 0)),
                 key=lambda x: x['mean_acc'], default=None)
if best_sweep and best_sweep['mean_acc'] > mean_hy:
    delta = best_sweep['mean_acc'] - mean_hy
    print(f"Best sweep config        : {best_sweep['label']}  (+{delta:.1%} vs baseline hybrid)")
else:
    print("No sweep config outperformed baseline hybrid.")
