# 02-preprocessing

Feature extraction and selection pipeline. Two complementary feature types feed the same downstream `train.py`.

| Pipeline | Script | Raw features | Selected |
|----------|--------|-------------|---------|
| **TSFresh** (default) | `feature_extraction.py` | ~774 | top 100 (MI) |
| **Topology** (opt-in) | `topology_extraction.py` | 24 | top 16 (MI) |

---

## Usage

### Via shell script (recommended)

```bash
# TSFresh only (default)
bash run-preprocessing.sh            # full dataset
bash run-preprocessing.sh test       # test dataset

# Topology only
bash run-preprocessing.sh test topo                   # sublevel+h1 (default)
bash run-preprocessing.sh test topo sublevel          # sublevel set only
bash run-preprocessing.sh test topo takens            # Takens embedding + Ripser
bash run-preprocessing.sh test topo both              # all methods combined
```

For hybrid training (TSFresh + topology merged), run both pipelines:

```bash
bash run-preprocessing.sh test        # TSFresh
bash run-preprocessing.sh test topo   # Topology
bash run-training.sh test hybrid
```

### Manual

```bash
cd 02-preprocessing

# --- TSFresh ---
python feature_extraction.py \
    --input  ../data/raw/dataset \
    --output ../data/features/dataset/allfeatures \
    --feature-set efficient \
    --n-jobs -1

python remove_leakage_features.py \
    --input ../data/features/dataset/allfeatures \
    --no-backup

python feature_selection.py \
    --input      ../data/features/dataset/allfeatures \
    --output     ../data/features/dataset/selected \
    --n-features 100 \
    --method     mutual_info \
    --target     primary

# --- Topology ---
python topology_extraction.py \
    --input  ../data/raw/dataset \
    --output ../data/features/dataset/topological \
    --method sublevel+h1 \
    --n-jobs -1

python feature_selection.py \
    --input      ../data/features/dataset/topological \
    --output     ../data/features/dataset/topological_selected \
    --n-features 16 \
    --method     mutual_info \
    --target     primary
```

---

## Scripts

### `feature_extraction.py` — TSFresh

Parallel feature extraction using TSFresh `EfficientFCParameters`.

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | — | Directory with raw parquet file(s) |
| `--output` | — | Output directory |
| `--feature-set` | `efficient` | `minimal` / `efficient` / `comprehensive` |
| `--n-jobs` | `0` | Parallel jobs (`-1` = all cores) |

Feature counts by set:

| Set | Features | Use case |
|-----|----------|----------|
| `minimal` | ~10 | Debugging |
| `efficient` | ~774 | Default |
| `comprehensive` | ~1200+ | Max features |

---

### `topology_extraction.py` — Persistent Homology

Extracts topological features from time series via persistent homology. Two filtration approaches are available, controlled by `--method`.

#### Methods

| Method | Diagrams | Features | Requires Ripser |
|--------|----------|----------|----------------|
| `sublevel+h1` | sub_H0 + sup_H0 + H1 | **24** | yes (H1 only) |
| `sublevel` | sub_H0 + sup_H0 | 16 | no |
| `takens` | H0 + H1 | 16 | yes |
| `both` | sub_H0 + sup_H0 + H0 + H1 | 32 | yes |

**Default: `sublevel+h1`** — three non-overlapping diagrams, each capturing different signal structure:

- **sub_H0** (sublevel set, H0): filtration `{f ≤ t}` from below.
  Births at local minima, deaths at local maxima (elder rule).
  Captures mean shifts (long bar at the shift level), valley depth, trend structure.

- **sup_H0** (superlevel set, H0): filtration `{f ≥ t}` from above.
  Equivalent to sublevel persistence of −f.
  Captures point anomalies (isolated spike → very high isolated bar), peaks, collective anomalies.

- **H1** (Takens + Vietoris-Rips): loops in the Takens embedding space.
  1D series → R^d point cloud → Ripser.
  Captures periodicity and oscillation structure — information unavailable from sublevel set alone (path complex has no loops).

Note: Takens H0 is excluded from `sublevel+h1` because it overlaps with sub/sup H0 — both capture level/clustering structure.

#### Per-diagram vectorisation (8 features each)

Each persistence diagram is summarised into 8 features:

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| `carl_f1` | Σ bᵢ(dᵢ−bᵢ) | birth-weighted persistence |
| `carl_f2` | Σ (d_max−dᵢ)(dᵢ−bᵢ) | death-proximity-weighted persistence |
| `carl_f3` | Σ bᵢ²(dᵢ−bᵢ)⁴ | high-degree birth term |
| `carl_f4` | Σ (d_max−dᵢ)²(dᵢ−bᵢ)⁴ | high-degree death term |
| `carl_f5_max` | max(dᵢ−bᵢ) | most persistent bar |
| `entropy` | −Σ(pᵢ/L)log(pᵢ/L) | diagram complexity (low = few dominant bars) |
| `landscape_l1` | Σ\|λ(x)\|dx | total topological signal magnitude |
| `landscape_l2` | √(Σλ(x)²dx) | dominant topological signal |

Carlsson coordinates f1–f5 are provably stable under Wasserstein perturbations of the diagram.

#### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--method` | `sublevel+h1` | Filtration method (see table above) |
| `--auto-delay` | off | Estimate Takens delay τ via Mutual Information (MI) |
| `--auto-dim` | off | Estimate Takens dimension d via False Nearest Neighbors (FNN) |
| `--embedding-delay` | `1` | Fixed τ (used when `--auto-delay` is off) |
| `--embedding-dim` | `3` | Fixed d (used when `--auto-dim` is off) |
| `--n-perm` | `500` | Subsample point cloud before Ripser (speeds up H1 computation) |
| `--homology-dims` | `0,1` | Homology dimensions (takens/both only) |
| `--n-jobs` | `-1` | Parallel jobs |

#### Takens parameter selection (takens/both/sublevel+h1 only)

| Method | Flag | Algorithm |
|--------|------|-----------|
| **MI delay** | `--auto-delay` | First local minimum of MI(x(t), x(t+τ)) — Fraser & Swinney 1986 |
| **FNN dimension** | `--auto-dim` | First d where false-neighbor fraction < 10% — Kennel et al. 1992 |

---

### `feature_selection.py` — Mutual Information

Selects top N features using mutual information. Selector is **fit on the training split only** to prevent data leakage.

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | — | Directory with `features.parquet` + `labels.parquet` |
| `--output` | — | Output directory |
| `--n-features` | `100` | Number of features to keep |
| `--method` | `mutual_info` | `variance` / `mutual_info` / `statistical` / `importance` |
| `--target` | `binary` | `primary` / `binary` / `sub` |
| `--test-size` | `0.2` | Fraction held out (must match `train.py`) |
| `--random-state` | `42` | Random seed (must match `train.py`) |

---

### `remove_leakage_features.py`

Removes TSFresh features that directly encode stationarity labels (ADF, KPSS, PP test statistics). Applied only to TSFresh output — not needed for topology.

Removed patterns: `adf`, `kpss`, `kwiatkowski`, `phillips_perron`, `unit_root`, `stationarity_test`, `variance_ratio_test`, `zivot_andrews`

---

## Output Structure

```
data/features/
├── test/  (or dataset/)
│   ├── allfeatures/                   # Raw TSFresh output
│   │   ├── features.parquet           # ~774 features × N series
│   │   ├── labels.parquet
│   │   └── feature_names.txt
│   ├── selected/                      # TSFresh after leakage removal + MI selection
│   │   └── primary/
│   │       ├── features.parquet       # 100 features × N series
│   │       └── labels.parquet
│   ├── topological/                   # Raw topology output
│   │   ├── features.parquet           # 24 features × N series (default)
│   │   ├── labels.parquet
│   │   └── feature_names.txt
│   └── topological_selected/          # Topology after MI selection
│       └── primary/
│           ├── features.parquet       # top features × N series
│           └── labels.parquet
```

Both `selected/primary/` and `topological_selected/primary/` feed directly into `train.py`.
