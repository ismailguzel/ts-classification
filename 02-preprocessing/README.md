# Feature Extraction & Preprocessing

This directory contains scripts for extracting statistical features from time series data and removing data leakage.

## 🚀 Recommended Workflow

We recommend using **`feature_extraction.py`** which is optimized for high-core servers using a **chunked file-based approach**. It provides immediate feedback and robust error handling.

### 1. Extract Features
This script reads Parquet files sequentially but processes each file using all available CPU cores in parallel.

```bash
python feature_extraction.py \
    --input ../data/raw/unified-20k \
    --output ../data/features/unified-20k/allfeatures \
    --feature-set efficient \
    --n-jobs 0  # 0 = Auto-detect all cores
```

### 2. Remove Leakage Features
Remove features that might leak stationarity information (e.g., ADF statistics).
```bash
python remove_leakage_features.py \
    --input ../data/features/unified-20k/allfeatures \
    --no-backup
```

### 3. Feature Selection (Optional)
Select the most relevant features for classification.
```bash
python feature_selection.py \
    --input ../data/features/unified-20k/allfeatures \
    --output ../data/features/unified-20k/selected \
    --n-features 100 \
    --method mutual_info \
    --target all
```

---

## 🛠️ Script Overview

| Script | Description | Usage |
|:---|:---|:---|
| **`feature_extraction.py`** | **Primary Tool.** Extracts TSFresh features using chunked processing. Robust to memory issues and allows monitoring progress file-by-file. | Main feature extraction step. |
| **`inspect_nans.py`** | **Diagnostics Tool.** Scans for NaNs/Infs and inspects specific series. | Debugging data quality issues. |
| **`remove_leakage_features.py`** | Removes stationarity test features (ADF, KPSS, etc.) to prevent data leakage. | Post-extraction cleanup. |
| **`feature_selection.py`** | Selects top-k features using Random Forest importance. | Reducing dimensionality. |

---

## 🔍 Debugging Data Quality

If `feature_extraction.py` reports bad rows, use `inspect_nans.py` to investigate.

**Scan all files:**
```bash
python inspect_nans.py scan --input ../data/raw/unified-20k --output nan_report.csv
```

**Inspect a specific series:**
```bash
python inspect_nans.py inspect --file path/to/file.parquet --series-id 15519
```

---

## ⚙️ Feature Set Options

| Feature Set | Count | Description |
|:---|:---:|:---|
| **`minimal`** | ~10 | Basic stats (mean, max, min, std). Fast testing. |
| **`efficient`** | **~780** | **(Recommended)** Standard set for classification. |
| **`comprehensive`** | ~1200+ | All features (slow, high compute). |

---

## 📂 Output Structure

- `features.parquet`: Extracted feature matrix (Series ID x Features).
- `labels.parquet`: Corresponding labels.
- `feature_names.txt`: List of extracted feature names.
