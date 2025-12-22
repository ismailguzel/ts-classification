# Feature Extraction & Preprocessing

This directory contains scripts for extracting features from time series data using multiple methods: statistical (TSFresh), topological (Persistent Homology), and their combination.

## Overview

The preprocessing pipeline supports three feature extraction approaches:

1. **Statistical Features** - Time-domain features using TSFresh (~800 raw → 100 selected)
2. **Topological Features** - Shape analysis using persistent homology (~200 raw → 50 selected)
3. **Combined Features** - Merge of statistical + topological (~1000 raw → 150 selected)

All pipelines are automated via `run-preprocessing.sh`.

## Automated Pipeline

```bash
# Run full preprocessing (all 3 feature types)
bash run-preprocessing.sh

# Or from project root
bash run.sh preprocessing
```

**Pipeline Steps:**
1. Statistical feature extraction (TSFresh)
2. Leakage removal (remove stationarity test features)
3. Statistical feature selection (Mutual Information)
4. Topological feature extraction (Ripser + Persistence Landscapes)
5. Topological feature selection
6. Feature combination (statistical + topological)
7. Combined feature selection

## Manual Usage

### 1. Statistical Features (TSFresh)

Extract time-domain statistical features:

```bash
python feature_extraction.py \
    --input ../data/raw/unified-20k \
    --output ../data/features/unified-20k/statistical \
    --feature-set efficient \
    --n-jobs 0  # Auto-detect cores
```

Remove leakage and select features:

```bash
# Remove stationarity test features
python remove_leakage_features.py \
    --input ../data/features/unified-20k/statistical \
    --no-backup

# Select top 100 features
python feature_selection.py \
    --input ../data/features/unified-20k/statistical \
    --output ../data/features/unified-20k/statistical_selected \
    --n-features 100 \
    --method mutual_info
```

### 2. Topological Features (Persistent Homology)

Extract topological features using Takens embedding and Ripser:

```bash
# Activate ts-top environment (required for ripser)
conda activate ts-top

python extract_topo_features.py \
    --input ../data/raw/unified-20k \
    --output ../data/features/unified-20k/topological \
    --n-jobs 16 \
    --use-mean-landscape  # Reduces features by 80%

# Select top 50 features
python feature_selection.py \
    --input ../data/features/unified-20k/topological \
    --output ../data/features/unified-20k/topological_selected \
    --n-features 50 \
    --method mutual_info
```

**Note:** Topological extraction requires `ts-top` conda environment with ripser installed.

### 3. Combined Features

Merge statistical and topological features:

```bash
python combine_features.py \
    --stat-dir ../data/features/unified-20k/statistical \
    --topo-dir ../data/features/unified-20k/topological \
    --output-dir ../data/features/unified-20k/combined

# Select top 150 features
python feature_selection.py \
    --input ../data/features/unified-20k/combined \
    --output ../data/features/unified-20k/combined_selected \
    --n-features 150 \
    --method mutual_info
```

---

## Script Overview

| Script | Purpose | Method | Output |
|:-------|:--------|:-------|:-------|
| **`feature_extraction.py`** | Extract statistical features | TSFresh (efficient set) | ~800 features |
| **`extract_topo_features.py`** | Extract topological features | Takens + Ripser + Landscapes | 200 features (mean mode) |
| **`combine_features.py`** | Merge statistical + topological | Horizontal concatenation | ~1000 features |
| **`remove_leakage_features.py`** | Remove stationarity test features | Pattern matching (ADF, KPSS, etc.) | Cleaned features |
| **`feature_selection.py`** | Select top-k features | Mutual Information | Selected subset |

## Feature Types Comparison

| Aspect | Statistical | Topological | Combined |
|:-------|:-----------|:------------|:---------|
| **Method** | TSFresh | Persistent Homology | Merge both |
| **Raw Features** | ~800 | 200 | ~1000 |
| **Selected** | 100 | 50 | 150 |
| **Captures** | Time-domain patterns | Shape & structure | Complementary info |
| **Examples** | Mean, autocorrelation, FFT | H0/H1 persistence, landscapes | Both types |
| **Environment** | ts-sktime | ts-top | ts-sktime |
| **Extraction Time** | ~20 min | ~30 min | ~50 min |

---

## Feature Set Options (Statistical)

| Feature Set | Count | Description | Use Case |
|:------------|:-----:|:------------|:---------|
| **`minimal`** | ~10 | Basic statistics (mean, std, min, max) | Quick testing |
| **`efficient`** | **~780** | **Recommended** - Standard classification set | Production |
| **`comprehensive`** | ~1200+ | All TSFresh features | Research/analysis |

## Topological Features Options

| Option | Features | Description |
|:-------|:--------:|:------------|
| **Normal mode** | 1000 | 5 landscapes × 200 statistics |
| **Mean landscape** | **200** | **Recommended** - Average across landscapes |

**Mean landscape mode reduces features by 80% while maintaining performance.**

## Output Structure

After preprocessing, features are organized as:

```
data/features/unified-20k/
├── statistical/              # Raw statistical features
│   ├── features.parquet
│   ├── labels.parquet
│   └── feature_names.txt
├── statistical_selected/     # Selected statistical features (100)
│   ├── features.parquet
│   └── labels.parquet
├── topological/              # Raw topological features
│   ├── features.parquet
│   ├── labels.parquet
│   └── feature_names.txt
├── topological_selected/     # Selected topological features (50)
│   ├── features.parquet
│   └── labels.parquet
├── combined/                 # Raw combined features
│   ├── features.parquet
│   ├── labels.parquet
│   └── feature_names.txt
└── combined_selected/        # Selected combined features (150)
    ├── features.parquet
    └── labels.parquet
```

## Environment Requirements

- **ts-sktime:** Statistical features, feature selection, training
- **ts-top:** Topological features (ripser library required)
  - Automatically activated by `run-preprocessing.sh` for Step 4

## Performance Notes

- **Statistical extraction:** ~20 minutes (20k samples, 64 cores)
- **Topological extraction:** ~30 minutes (20k samples, 16 cores, mean landscape mode)
- **Memory usage:** 16-32GB recommended for 20k dataset
- **Disk space:** ~2-3GB per feature type

## See Also

- **Topological methods:** `topological-features/README.md`
- **Model training:** `../03-models/hierarchical/README.md`
- **Results analysis:** `../04-postprocessing/README.md`
