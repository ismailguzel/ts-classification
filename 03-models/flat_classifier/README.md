# Flat 39-Class Classifier

Trains four classifiers on extracted features and evaluates on a held-out test split.

---

## Usage

### Via shell script (recommended)

```bash
# TSFresh features (default)
bash run-training.sh            # full dataset
bash run-training.sh test       # test dataset

# Topology features only
bash run-training.sh test topo

# Hybrid (TSFresh + topology merged)
bash run-training.sh test hybrid
```

### Manual

```bash
cd 03-models/flat_classifier

# TSFresh features
python train.py \
    --features-path ../../data/features/dataset/selected \
    --save-dir ./output/shape_tsfresh

# Topology features
python train.py \
    --features-path ../../data/features/dataset/topological_selected \
    --save-dir ./output/shape_topo

# Hybrid (TSFresh + topology merged on series_id)
python train.py \
    --features-path       ../../data/features/dataset/selected \
    --extra-features-path ../../data/features/dataset/topological_selected \
    --save-dir            ./output/shape_hybrid

# Specific classifier only
python train.py --features-path ... --classifier xgboost
```

---

## Feature Modes

| Mode | Flag | Input | Output dir |
|------|------|-------|-----------|
| TSFresh | (default) | `data/features/<mode>/selected` | `output/<mode>_tsfresh/` |
| Topology | `topo` | `data/features/<mode>/topological_selected` | `output/<mode>_topo/` |
| Hybrid | `hybrid` | both merged on series_id | `output/<mode>_hybrid/` |

The hybrid mode performs an inner join of TSFresh and topology features on `series_id`, so only series present in both sets are used.

---

## Classifiers

| Name | Algorithm | Notes |
|------|-----------|-------|
| `rf` | Random Forest | 200 estimators |
| `xgboost` | XGBoost | 200 estimators, mlogloss |
| `catboost` | CatBoost | 200 iterations, depth 6 |
| `svm` | SVM RBF | StandardScaler applied |

`--classifier all` (default) trains all four and saves the best by test accuracy.

All classifiers use the same stratified 80/20 train/test split. `StandardScaler` is fit on the training split only.

---

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--features-path` | `../../data/features/dataset/selected` | Primary feature directory |
| `--extra-features-path` | — | Second feature directory (hybrid mode) |
| `--classifier` | `all` | `all` / `rf` / `xgboost` / `catboost` / `svm` |
| `--n-jobs` | `-1` | Parallel jobs |
| `--test-size` | `0.2` | Train/test split ratio |
| `--random-state` | `42` | Random seed |
| `--save-dir` | `output` | Output directory. `run-training.sh` always passes `output/<mode>_<features>/`, so Study A and Study B runs cannot overwrite each other. |

---

## Output

```
03-models/flat_classifier/
└── output/
    ├── shape_tsfresh/           # one directory per <mode>_<features>,
    ├── shape_topo/              # so studies never overwrite each other
    ├── shape_hybrid/
    ├── season-structure_topo/
    ├── season-anomaly_topo/
    └── <mode>_<features>/
        ├── classifier.pkl           # Best model (by test accuracy)
        ├── scaler.pkl               # Fitted StandardScaler
        ├── metadata.pkl             # Class names, feature names, paths
        ├── summary.json             # Accuracy comparison + feature_type
        ├── metrics_RandomForest.json
        ├── metrics_XGBoost.json
        ├── metrics_CatBoost.json
        ├── metrics_SVM_RBF.json
        ├── feature_importance_*.csv # Per-model importances
        ├── predictions_*.csv        # Per-sample true/predicted labels
        └── misclassified_*.csv      # Misclassified samples with confidence
```
