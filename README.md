# Hierarchical Time Series Classification for Stationarity

A machine learning pipeline to classify time series stationarity using a hierarchical approach with multiple feature extraction methods.

- **Level 1:** Binary Classification (Stationary vs. Non-Stationary)
- **Level 2:** Multi-class Classification (Trend, Volatility, Stochastic, Anomaly, Structural Break)

## Quick Start

### Full Pipeline (20k Dataset)

```bash
# Default: 20k dataset, statistical features
bash run.sh

# With specific feature type
bash run.sh all 20k statistical   # Statistical features (TSFresh)
bash run.sh all 20k topological   # Topological features (Persistent Homology)
bash run.sh all 20k combined      # Combined features (Statistical + Topological)
```

### Fast Testing (5k Dataset)

```bash
# Run full pipeline with all 3 feature types on 5k dataset
bash run-test-5k.sh

# Test specific steps
bash run-test-5k.sh generation      # Generate 5k dataset only
bash run-test-5k.sh preprocessing   # Extract features only
bash run-test-5k.sh training        # Train all 3 feature types
```

### Step-by-Step Execution

```bash
# Individual pipeline steps
bash run.sh generation          # 1. Generate synthetic data
bash run.sh preprocessing       # 2. Extract & select features
bash run.sh training            # 3. Train models
bash run.sh postprocessing      # 4. Analyze & visualize results
bash run.sh baseline            # 5. Compare with traditional tests

# Or use direct scripts
bash run-generation.sh
bash run-preprocessing.sh
bash run-training.sh
bash run-postprocessing.sh statistical 20k
bash run-baseline.sh
```

### Cleanup

```bash
# Remove all generated files
bash clean.sh all

# Clean specific parts
bash clean.sh data      # Only raw data
bash clean.sh features  # Only extracted features
bash clean.sh models    # Only trained models
bash clean.sh logs      # Only log files
```

---

## Project Structure

```
.
├── 01-data-generation/          # Synthetic data generation
│   ├── config.py                # Dataset configuration
│   └── generate.py              # Main generation script
├── 02-preprocessing/            # Feature extraction & selection
│   ├── feature_extraction.py        # Statistical features (TSFresh)
│   ├── extract_topo_features.py     # Topological features (Ripser)
│   ├── combine_features.py          # Merge statistical + topological
│   ├── feature_selection.py         # Mutual Information selection
│   ├── remove_leakage_features.py   # Remove stationarity test features
│   └── topological-features/        # Topological computation core
│       ├── topological_features.py  # Takens + Ripser + Landscapes
│       └── README.md
├── 03-models/                   # Hierarchical classification models
│   ├── hierarchical/
│   │   ├── model1_binary/           # Binary classifier (Stationary/Non-Stationary)
│   │   └── model2_nonstationary/    # 5-class classifier (Pattern types)
│   └── utils/                       # Shared utilities
├── 04-postprocessing/           # Results analysis & visualization
│   ├── visualize_feature_importance.py
│   ├── visualize_errors_simple.py
│   └── figures/                     # Generated plots
├── 05-baseline-comparison/      # Traditional statistical tests (ADF, KPSS, PP)
├── data/                        # Generated datasets
│   ├── raw/                         # Time series (unified-5k, unified-20k)
│   └── features/                    # Extracted features (statistical, topological, combined)
├── logs/                        # Pipeline execution logs
├── run.sh                       # Master pipeline script
├── run-test-5k.sh               # Fast testing with 5k dataset
├── clean.sh                     # Cleanup script
└── requirements.txt             # Python dependencies
```

## Key Features

### Hierarchical Architecture
- **Two-stage classification:** Binary (Stationary vs. Non-Stationary) → Multi-class (Pattern types)
- **Flexible training:** Train models independently or in sequence
- **Comprehensive evaluation:** Per-classifier metrics, confusion matrices, feature importance

### Multiple Feature Extraction Methods

| Feature Type | Method | Raw Features | Selected | Use Case |
|:-------------|:-------|:------------:|:--------:|:---------|
| **Statistical** | TSFresh | ~800 | 100 | Time-domain statistical properties |
| **Topological** | Persistent Homology | 200 | 50 | Shape and structure analysis via Takens embedding |
| **Combined** | Statistical + Topological | ~1000 | 150 | Complementary information from both methods |

**Feature Extraction Details:**
- **Statistical Features:** Mean, variance, autocorrelation, FFT coefficients, entropy, trend analysis
- **Topological Features:** Persistent homology (H0/H1) via Ripser, persistence landscapes, topological entropy
- **Selection Method:** Mutual Information for all types
- **Leakage Prevention:** Automatic removal of stationarity test features (ADF, KPSS statistics)

### Performance & Scalability
- **Chunked processing:** Handles large datasets (20k+ samples) with memory constraints
- **Parallel execution:** Multi-core feature extraction and model training
- **Fast testing:** 5k dataset for rapid prototyping (~30-60 minutes)
- **Full pipeline:** 20k dataset for production results (~2-4 hours)

## Dataset Categories

The synthetic dataset includes **6 primary categories** covering diverse time series patterns:

### Statistical Features (Current Results)

| Model | Accuracy | F1-Score | Features | Classifiers |
|:------|:--------:|:--------:|:--------:|:------------|
| **Model 1** (Binary) | **96.89%** | 96.88% | 100 | XGBoost, CatBoost, LightGBM |
| **Model 2** (5-Class) | **97.81%** | 97.78% | 100 | XGBoost, CatBoost, LightGBM |

### Baseline Comparison (Traditional Tests)

| Method | Accuracy | Approach |
|:-------|:--------:|:---------|
| **Hierarchical ML** | **96.89%** | TSFresh features + XGBoost |
| KPSS Test | 79.14% | Statistical test (constant trend) |
| ADF Test | 73.03% | Statistical test (unit root) |
| Phillips-Perron | 65.41% | Statistical test (unit root) |

**Improvement:** +17.75% absolute over best baseline (KPSS)

### Feature Type Comparison

| Feature Type | Model 1 Accuracy | Model 2 Accuracy | Training Time | Notes |
|:-------------|:----------------:|:----------------:|:-------------:|:------|
| Statistical | 96.89% | 97.81% | ~30 min | Baseline, well-established |
| Topological | TBD | TBD | ~45 min | Experimental, shape analysis |
| Combined | TBD | TBD | ~60 min | Potential performance boost |

*Note: Topological and combined features are under evaluation*
**Hierarchical Mapping:**
- **Level 1 (Binary):** Stationary (1 category) vs Non-Stationary (5 categories)
### Python Environment
- **Python:** 3.8+
- **RAM:** 16GB+ (recommended 32GB+ for 20k dataset)
- **Storage:** ~5GB for full pipeline outputs

### Dependencies

```bash
pip install -r requirements.txt
```

**Core packages:**
- `pandas`, `numpy`, `scikit-learn` - Data processing & ML
- `tsfresh` - Statistical feature extraction
- `xgboost`, `catboost`, `lightgbm` - Gradient boosting classifiers
- `ripser` - Topological feature extraction (persistent homology)
- `matplotlib`, `seaborn` - Visualization

**Conda Environments:**
- `ts-sktime` - Main environment (statistical features, training)
- `ts-top` - Topological features (ripser dependencies)
  - *Automatically activated during topological extraction*

## Documentation

- **Preprocessing:** `02-preprocessing/README.md` - Feature extraction details
- **Topological Features:** `02-preprocessing/topological-features/README.md` - Persistent homology methods
- **Model Training:** `03-models/hierarchical/*/README.md` - Model architecture and usage
- **Baseline Comparison:** `05-baseline-comparison/README.md` - Traditional statistical tests
- **Technical Report:** `TECHNICAL_REPORT_19-12-2025.pdf` - Detailed methodology and results

## Results Structure

After running the pipeline, results are organized by dataset size and feature type:

```
data/
├── raw/unified-20k/              # Generated time series
├── features/unified-20k/
│   ├── statistical_selected/     # 100 selected statistical features
│   ├── topological_selected/     # 50 selected topological features
│   └── combined_selected/        # 150 selected combined features

03-models/hierarchical/
├── model1_binary/output/
│   ├── model1_binary_features/   # Statistical (20k, default)
│   ├── model1_topological/       # Topological (20k)
│   ├── model1_combined/          # Combined (20k)
│   ├── model1_statistical_5k/    # Statistical (5k test)
│   ├── model1_topological_5k/    # Topological (5k test)
│   └── model1_combined_5k/       # Combined (5k test)

04-postprocessing/figures/
├── statistical/                  # Feature importance, error analysis (20k)
├── topological/
├── combined/
├── statistical_5k/               # Figures for 5k test runs
├── topological_5k/
└── combined_5k/
```

## Citation

If you use this work, please cite:

```
Hierarchical Time Series Classification for Stationarity Detection
Technical Report, December 2025
Dataset: 20,000 synthetic time series samples
```

## License

This project is for research and educational purpose
| **Model 1** (Binary) | Topological | **TBD** | 50 | Using persistent homology features |
| **Model 2** (5-Class) | Topological | **TBD** | 50 | Using persistent homology features |
| **Model 1** (Binary) | Combined | **TBD** | 150 | Statistical + Topological |
| **Model 2** (5-Class) | Combined | **TBD** | 150 | Statistical + Topological |

**Note**: Topological and combined features are experimental and currently being evaluated.

## Requirements

- Python 3.8+
- Dependencies: `pandas`, `numpy`, `scikit-learn`, `tsfresh`, `xgboost`, `joblib`
- Recommended: 64GB+ RAM for feature extraction (or use chunking).

## Documentation

- **Preprocessing:** See `02-preprocessing/README.md` for details on feature sets and diagnostics.
- **Baselines:** See `05-baseline-comparison/README.md` for comparison with traditional statistical tests.
