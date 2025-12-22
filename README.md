# Hierarchical Time Series Classification for Stationarity

A machine learning pipeline to classify time series stationarity using a hierarchical approach.
- **Level 1:** Binary Classification (Stationary vs. Non-Stationary)
- **Level 2:** Multi-class Classification (Trend, Volatility, Stochastic, Anomaly, Structural Break)

## Quick Start

The pipeline is fully automated via the master script `run.sh`.

### 1. Full Pipeline
Runs everything: Generation -> Preprocessing -> Training -> Post-Processing -> Baseline.
```bash
bash run.sh
```

### 2. Step-by-Step Execution
You can also run individual steps:

```bash
# 1. Data Generation
bash run.sh generation
# Or: bash run-generation.sh

# 2. Preprocessing (Feature Extraction & Selection)
bash run.sh preprocessing
# Or: bash run-preprocessing.sh

# 3. Model Training
bash run.sh training
# Or: bash run-training.sh

# 4. Post-Processing (Analysis & Visualization)
bash run.sh postprocessing
# Or: bash run-postprocessing.sh

# 5. Baseline Comparison
bash run.sh baseline
# Or: bash run-baseline.sh
```

---

## Project Structure

```
.
├── 01-data-generation/      # Synthetic data generation scripts
├── 02-preprocessing/        # Feature extraction & selection
│   ├── feature_extraction.py   # Optimized TSFresh extractor (Pandas/Chunked)
│   ├── remove_leakage_features.py
│   ├── feature_selection.py
│   └── inspect_nans.py         # Data quality diagnostics
├── 03-models/               # Hierarchical classification models
│   ├── hierarchical/model1_binary/
│   └── hierarchical/model2_nonstationary/
├── 04-postprocessing/       # Results analysis & visualization
├── 05-baseline-comparison/  # Comparison with traditional tests (ADF, KPSS)
├── run.sh                   # Master pipeline script
├── run-generation.sh        # Automation script for Step 1
├── run-preprocessing.sh     # Automation script for Step 2
├── run-training.sh          # Automation script for Step 3
├── run-postprocessing.sh    # Automation script for Step 4
└── run-baseline.sh          # Automation script for Step 5
```

## Key Features

- **Hierarchical Classification:** Decomposes the problem into binary and multi-class steps.
- **Optimized Feature Extraction:** Uses a chunked, parallelized Pandas approach (`feature_extraction.py`) for high-performance feature generation on large datasets.
- **Data Leakage Prevention:** Automatically removes stationarity-related features (like ADF statistics) to ensure fair model evaluation.
- **Feature Selection:** Selects the top 100 most relevant features to improve model speed and accuracy.

## Dataset Categories

The synthetic dataset includes **6 primary categories** covering diverse time series patterns:

| Category | Count | Percentage | Description |
|:---------|------:|:----------:|:------------|
| **stationary** | 9,988 | 52.2% | AR, MA, ARMA, White Noise processes |
| **trend** | 1,960 | 10.2% | Linear, Quadratic, Cubic, Exponential, Damped trends |
| **volatility** | 1,996 | 10.4% | ARCH, GARCH, EGARCH, APARCH models |
| **stochastic** | 1,995 | 10.4% | Random Walk, RW with Drift, ARI, IMA, ARIMA |
| **anomaly** | 1,548 | 8.1% | Point Anomalies (Single/Multiple), Collective Anomalies |
| **structural_break** | 1,660 | 8.7% | Mean Shift, Variance Shift, Trend Shift |
| **Total** | **19,147** | **100%** | 20K dataset (unified-20k) |

**Hierarchical Mapping:**
- **Level 1 (Binary):** Stationary (1 category) vs Non-Stationary (5 categories)
- **Level 2 (5-Class):** Trend, Volatility, Stochastic, Anomaly, Structural Break

## Performance

| Model | Accuracy | Description |
|:---|:---:|:---|
| **Model 1** (Binary) | **96.89%** | Distinguishes Stationary vs. Non-Stationary |
| **Model 2** (5-Class) | **97.81%** | Classifies Non-Stationary types (Trend, Volatility, etc.) |

## Requirements

- Python 3.8+
- Dependencies: `pandas`, `numpy`, `scikit-learn`, `tsfresh`, `xgboost`, `joblib`
- Recommended: 64GB+ RAM for feature extraction (or use chunking).

## Documentation

- **Preprocessing:** See `02-preprocessing/README.md` for details on feature sets and diagnostics.
- **Baselines:** See `05-baseline-comparison/README.md` for comparison with traditional statistical tests.
