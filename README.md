# Hierarchical Time Series Classification

A comprehensive pipeline for hierarchical classification of time series stationarity patterns using synthetic data generation, TSFresh feature engineering, and advanced machine learning models.

## Project Overview

This project implements a two-stage hierarchical classification system for time series stationarity detection:

```
Input Time Series
    ↓
[Model 1: Binary Classification]
    ├─→ Stationary (0) ✓
    └─→ Non-Stationary (1)
            ↓
    [Model 2: 5-Class Classification]
        ├─→ Trend (0)
        ├─→ Stochastic (1)
        ├─→ Volatility (2)
        ├─→ Anomaly (3)
        └─→ Structural Break (4)
```

### Key Features
- **Scalable Synthetic Data Generation**: 1.5K to 200K samples with 11 configurable presets
- **Three Training Modes**: RAW (sktime), FEATURES (sklearn), AutoTrain (AutoGluon)
- **State-of-the-Art Performance**: 96.55% (Model 1) and 97.92% (Model 2) accuracy
- **Comprehensive Pipeline**: Data generation → Feature extraction → Model training → Visualization
- **Publication-Ready Visualizations**: Error analysis and feature importance figures

---

## Project Structure

```
hierarchical-ts-classification/
├── 01-data-generation/          # Synthetic time series generation
│   ├── generate.py              # Main generation script (11 scale presets)
│   ├── verify_ids.py            # Data integrity verification
│   └── config.py                # Configuration management
│
├── 02-preprocessing/            # Feature engineering (OPTIONAL for RAW mode)
│   ├── extract_dask.py          # TSFresh feature extraction (Dask parallel)
│   ├── feature_selection.py    # Mutual information feature selection
│   └── topological-features/    # Experimental TDA-based features
│
├── 03-models/hierarchical/      # Hierarchical classification models
│   ├── model1_binary/           # Binary classifier (Stationary vs Non-Stationary)
│   │   ├── train_model1.py      # Training script (RAW/FEATURES modes)
│   │   ├── autotrain_models1.py # AutoGluon training
│   │   └── saved_models/        # Trained models and results
│   │
│   ├── model2_nonstationary/    # 5-class classifier (Non-stationary types)
│   │   ├── train_model2.py      # Training script (RAW/FEATURES modes)
│   │   ├── autotrain_models2.py # AutoGluon training
│   │   └── saved_models/        # Trained models and results
│   │
│   └── utils/                   # Shared utilities
│       ├── metrics.py           # ModelEvaluator class
│       ├── data_utils.py        # Data loading utilities
│       └── constants.py         # Label mappings
│
├── 04-postprocessing/           # Analysis and visualization
│   ├── visualize_errors_simple.py        # Error analysis plots
│   ├── visualize_feature_importance.py   # Feature importance plots
│   ├── visualize_misclassified.py        # Individual sample plots
│   └── figures/                          # Generated visualizations
│
├── data/                        # Data storage
│   ├── raw/                     # Raw time series (Parquet format)
│   └── features/                # TSFresh features (optional)
│
├── README.md                    # Project documentation
├── TECHNICAL_REPORT.md          # Comprehensive technical report
├── requirements.txt             # Python dependencies
├── smoke_test.sh                # Quick validation test
├── run-20k-trainig.sh           # Complete training pipeline
└── run-postprocessing.sh        # Visualization pipeline
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/ismailguzel/hierarchical-ts-classification.git
cd hierarchical-ts-classification
pip install -r requirements.txt
```

### 2. Generate Data

```bash
cd 01-data-generation

# Generate test dataset (recommended for quick testing)
python generate.py --scale test  # 1.5K samples

# Generate larger datasets (strategic progression)
python generate.py --scale 5k    # 5K samples
python generate.py --scale 10k   # 10K samples
python generate.py --scale 20k   # 20K samples
python generate.py --scale 90k   # 90K samples (baseline)
python generate.py --scale 200k  # 200K samples

# See all available scales
python config.py
```

### 3. Train Models

```bash
cd 03-models/hierarchical/model1_binary

# Model 1: Binary Classification (RAW mode - recommended)
python train_model1.py --mode raw --classifier rocket

# Model 2: 5-Class Classification
cd ../model2_nonstationary
python train_model2.py --mode raw --classifier rocket
```

### 4. Generate Visualizations

```bash
# Run post-processing pipeline to generate error analysis and feature importance figures
./run-postprocessing.sh
```

**Output**: 6 publication-quality figures in `04-postprocessing/figures/`:
- Error analysis for both models
- Feature importance visualizations (top-20 and category breakdowns)

### Alternative Workflows

**FEATURES Mode** (TSFresh + sklearn):
```bash
cd 02-preprocessing
python extract_dask.py \
    --input ../data/raw/unified-5k \
    --output ../data/features/unified-5k/allfeatures \
    --n-workers 4
python feature_selection.py \
    --input ../data/features/unified-5k/allfeatures \
    --output ../data/features/unified-5k/selected \
    --method mutual_info \
    --target all
cd ../03-models/hierarchical/model1_binary
python train_model1.py --mode features
```

**AutoTrain Mode** (AutoGluon):
```bash
# After feature extraction and selection
cd 03-models/hierarchical/model1_binary

# Using AutoGluon
python autotrain_models1.py \
    --features-path ../../../data/features/unified-5k/selected \
    --time-limit 3600 \
    --presets medium_quality_faster_train

# Model 2 AutoTrain
cd ../model2_nonstationary
python autotrain_models2.py \
    --features-path ../../../data/features/unified-5k/selected \
    --time-limit 3600 \
    --presets medium_quality_faster_train
```

**Note:** AutoTrain requires feature extraction first (see 02-preprocessing/).

**Smoke Test** (Quick Validation):
```bash
# Tests all modes with small dataset
bash smoke_test.sh
```

**Batch Training** (20K Dataset):
```bash
# Runs complete training pipeline
bash run-20k-trainig.sh
```

---

## Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DATA GENERATION (01-data-generation/)                       │
│    • generate.py → Scalable datasets (test/5k/10k/.../200k)   │
│    • config.py → 11 scale presets (1.5K to 200K)              │
│    • verify_ids.py → Data integrity check                      │
└─────────────────────────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. PREPROCESSING (02-preprocessing/) [OPTIONAL]                │
│    • extract_dask.py → TSFresh features (Dask parallel)        │
│    • feature_selection.py → Select relevant features           │
│    • topological-features/ → Experimental TDA features          │
│    └─→ Only for FEATURES/AutoTrain modes                       │
└─────────────────────────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. MODEL TRAINING (03-models/hierarchical/)                    │
│                                                                 │
│    A. RAW Mode (Recommended)                                   │
│       → Direct time series → sktime classifiers                │
│                                                                 │
│    B. FEATURES Mode                                            │
│       → TSFresh features → sklearn classifiers                 │
│                                                                 │
│    C. AutoTrain Mode                                           │
│       → TSFresh features → AutoML (AutoGluon)                  │
└─────────────────────────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. HIERARCHICAL CLASSIFICATION                                 │
│                                                                 │
│    Model 1 (Binary)                                            │
│    ├─→ Stationary (0) ✓                                        │
│    └─→ Non-Stationary (1)                                      │
│            ↓                                                    │
│    Model 2 (5-Class)                                           │
│    ├─→ Trend (0)                                               │
│    ├─→ Volatility (1)                                          │
│    ├─→ Stochastic (2)                                          │
│    ├─→ Anomaly (3)                                             │
│    └─→ Structural Break (4)                                    │
└─────────────────────────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. POST-PROCESSING & VISUALIZATION (04-postprocessing/)        │
│    • visualize_errors_simple.py → Error analysis figures       │
│    • visualize_feature_importance.py → Feature importance      │
│    • visualize_misclassified.py → Individual sample plots      │
│    └─→ 6 publication-quality figures + detailed reports        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Training Modes

| Mode | Input | Classifiers | When to Use |
|------|-------|-------------|-------------|
| **RAW** | Time series | sktime (ROCKET, Arsenal) | Quick start, baseline |
| **FEATURES** | TSFresh features | sklearn (XGBoost, SVM) | Feature analysis, interpretability |
| **AutoTrain** | TSFresh features | AutoML (AutoGluon) | Automated tuning |

See individual README files for detailed configuration options.

---

## Output Structure

### Training Outputs

Training scripts automatically organize outputs into mode-specific directories:

```
03-models/hierarchical/model1_binary/saved_models/
├── model1_binary_raw_rocket/
│   ├── model1_binary_classifier.pkl       # Trained model
│   ├── model1_features_summary.json       # Comprehensive metrics
│   ├── predictions.csv                    # All predictions
│   └── misclassified.csv                  # Error analysis
│
├── model1_binary_features/
│   └── [Same structure for FEATURES mode]
│
└── model1_binary_autogluon/
    ├── models/                            # AutoGluon ensemble
    └── [Same CSV/JSON outputs]
```

### Visualization Outputs

Post-processing pipeline generates publication-ready figures:

```
04-postprocessing/figures/
├── error_analysis_model_1_(binary_classification).png
├── error_analysis_model_2_(5-class_classification).png
├── feature_importance_top20_comparison_model1.png
├── feature_importance_categories_model1.png
├── feature_importance_top20_comparison_model2.png
└── feature_importance_categories_model2.png
```

---

## Requirements

```bash
pip install -r requirements.txt

# Optional: AutoGluon support
pip install autogluon.tabular
```

Core dependencies:
- `tsfresh` - Time series feature extraction
- `scikit-learn` - Traditional ML classifiers
- `sktime` - Time series classifiers (ROCKET, Arsenal)
- `pandas`, `numpy` - Data manipulation
- `dask` - Parallel feature extraction

---

## Performance Results

### 20K Dataset Benchmark

**Model 1 (Binary Classification)**:
- **Best Accuracy**: 96.55% (CatBoost)
- **Test Samples**: 5,328
- **Misclassifications**: 132 samples

**Model 2 (5-Class Classification)**:
- **Best Accuracy**: 97.92% (XGBoost)
- **Test Samples**: 3,996 (non-stationary only)
- **Misclassifications**: 38 samples

See [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) for detailed analysis, methodology, and visualizations.

---

## Documentation

### Component Documentation
- **Data Generation**: `01-data-generation/README.md`
- **Preprocessing**: `02-preprocessing/README.md`
- **Models Overview**: `03-models/hierarchical/README.md`
- **Model 1 Details**: `03-models/hierarchical/model1_binary/README.md`
- **Model 2 Details**: `03-models/hierarchical/model2_nonstationary/README.md`
- **Post-Processing**: `04-postprocessing/README.md`

### Reports
- **Technical Report**: `TECHNICAL_REPORT.md` - Comprehensive analysis with figures
- **Training Pipeline**: `TRAINING_PIPELINE.md` - Detailed workflow documentation
- **20K Training Report**: `TRAINING_REPORT_20K.md` - Benchmark results

---

## Complete Pipeline Example (20K Dataset)

### Step 1: Data Generation
```bash
cd 01-data-generation
python generate.py --scale 20k
```

**Output**: 19,980 synthetic time series samples in Parquet format

### Step 2: Feature Extraction (Optional - for FEATURES/AutoTrain modes)
```bash
cd 02-preprocessing

# Extract TSFresh features (~780 features per sample)
python extract_dask.py \
    --input ../data/raw/unified-20k \
    --output ../data/features/unified-20k/allfeatures \
    --n-workers 55

# Select top 100 features using mutual information
python feature_selection.py \
    --input ../data/features/unified-20k/allfeatures \
    --output ../data/features/unified-20k/selected \
    --method mutual_info \
    --n-features 100 \
    --target all
```

**Output**: Optimized feature sets for both Model 1 (binary) and Model 2 (5-class)


### Step 3: Model Training

**Option A: Quick Training (RAW mode - recommended)**
```bash
cd 03-models/hierarchical/model1_binary
python train_model1.py --mode raw

cd ../model2_nonstationary
python train_model2.py --mode raw
```

**Option B: Feature-Based Training**
```bash
cd 03-models/hierarchical/model1_binary
python train_model1.py --mode features --features-path ../../../data/features/unified-20k/selected

cd ../model2_nonstationary
python train_model2.py --mode features --features-path ../../../data/features/unified-20k/selected
```

**Option C: AutoML Training**
```bash
cd 03-models/hierarchical/model1_binary
python autotrain_models1.py --features-path ../../../data/features/unified-20k/selected --time-limit 3600

cd ../model2_nonstationary
python autotrain_models2.py --features-path ../../../data/features/unified-20k/selected --time-limit 3600
```

**Batch Processing**: Use `./run-20k-trainig.sh` to execute the complete training pipeline automatically.

### Step 4: Post-Processing & Visualization

```bash
# Generate error analysis and feature importance figures
./run-postprocessing.sh
```

**Output**: 6 figures in `04-postprocessing/figures/` with detailed analysis logs

---

## Key Technologies

- **TSFresh**: Time series feature extraction (780+ features)
- **sktime**: Time series classification (ROCKET, Arsenal)
- **scikit-learn**: Traditional ML classifiers (XGBoost, CatBoost, SVM, RF)
- **AutoGluon**: Automated machine learning
- **Dask**: Parallel feature extraction
- **Pandas/NumPy**: Data manipulation

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{guzel2025hierarchical,
  title={Hierarchical Time Series Classification},
  author={Guzel, Ismail},
  year={2025},
  url={https://github.com/ismailguzel/hierarchical-ts-classification}
}
```

---

## License

This project is licensed under the MIT License.

---

## Author

**Ismail Guzel**
- GitHub: [@ismailguzel](https://github.com/ismailguzel)

---

## References

- **TSFresh**: Christ, M., et al. (2018). Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests. *Neurocomputing*.
- **sktime**: Löning, M., et al. (2019). sktime: A Unified Interface for Machine Learning with Time Series. *NeurIPS Workshop*.
- **ROCKET**: Dempster, A., et al. (2020). ROCKET: Exceptionally fast and accurate time series classification. *Data Mining and Knowledge Discovery*.
- **AutoGluon**: Erickson, N., et al. (2020). AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data. *AutoML Workshop at ICML*.