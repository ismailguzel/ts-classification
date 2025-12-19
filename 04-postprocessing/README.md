# 04-postprocessing

Post-processing and analysis tools for trained models.

## Main Tools (Production Ready)

### 1. `visualize_confusion_matrices.py` - **NEW!** Confusion Matrix Generator

Generate high-quality confusion matrices for all trained classifiers.

**Features:**
- Create individual confusion matrices for each classifier
- Generate grid layouts showing all models together
- Support for both raw counts and normalized (proportions) matrices
- Large, clear annotations (fontsize=14, bold)
- Multiple output formats (PNG, PDF, SVG)
- Automatic class name labeling (no more 0, 1, 2...)

**Usage:**

```bash
# Generate all confusion matrices (both models, grid + individual)
python visualize_confusion_matrices.py --model both --save-fig

# Model 1 only with normalized values
python visualize_confusion_matrices.py --model model1 --normalize --save-fig

# Model 2, grid layout only, save as PDF
python visualize_confusion_matrices.py --model model2 --grid-only --save-fig --format pdf

# Individual plots only (no grid)
python visualize_confusion_matrices.py --model both --individual-only --save-fig
```

**Output:**
- `cm_model_1_binary_classification_<classifier>.png` - Individual confusion matrices for Model 1
- `cm_model_2_5-class_classification_<classifier>.png` - Individual confusion matrices for Model 2
- `cm_grid_model_1_binary_classification.png` - Grid layout for Model 1
- `cm_grid_model_2_5-class_classification.png` - Grid layout for Model 2

### 2. `visualize_errors_simple.py` - Batch Error Analysis

Analyze and visualize misclassification patterns across all classifiers.

**Features:**
- Load all misclassified samples from saved models
- Show error counts by classifier
- Display confusion matrix for errors
- Plot confidence distribution for misclassified samples
- List worst cases (highest confidence errors)

**Usage:**

```bash
# Basic usage - analyze Model 1 errors
python visualize_errors_simple.py --model model1

# Show top 20 worst cases, skip plots
python visualize_errors_simple.py --model model1 --top 20 --no-plot

# Analyze Model 2 and save figures
python visualize_errors_simple.py --model model2 --save-fig
```

**Output:**
- Error count by classifier
- Aggregated confusion matrix (errors only)
- Confidence distribution histogram
- Errors by true label
- Top N worst misclassifications with confidence scores

### 2. `visualize_misclassified.py` - Individual Time Series Visualization

Visualize individual misclassified time series with prediction details and statistical properties.

**Features:**
- Load time series by `series_id` from raw parquet files
- Display true label vs predicted label with confidence
- Plot time series with trend line, mean, and statistical annotations
- Show all prediction probabilities for each class
- Calculate statistical properties (mean, std, trend slope, etc.)
- Save high-resolution figures (150 dpi PNG)

**Usage:**

```bash
# Visualize a misclassified sample
python visualize_misclassified.py --id 15255 --model model1

# Save figure to figures/ directory
python visualize_misclassified.py --id 15255 --model model1 --save-fig

# Use Model 2 (5-class classification)
python visualize_misclassified.py --id 17071 --model model2 --save-fig

# Custom paths
python visualize_misclassified.py --id 3344 --model model1 \
    --saved-models-dir ../03-models/hierarchical/model1_binary/saved_models/model1_binary_features \
    --raw-data-dir ../data/raw/unified-20k
```

**How to Find Series IDs:**
1. Check misclassified CSV files for interesting errors
2. Use `visualize_errors_simple.py` to see worst cases with IDs
3. IDs are preserved throughout the pipeline: generation → features → predictions

**Note:** The `id` column in predictions/misclassified CSV files is the actual `series_id` from data generation, so you can directly use those IDs.

### 3. `visualize_feature_importance.py` - Feature Importance Analysis

Analyze and visualize feature importance across classifiers with automatic categorization.

**Features:**
- Load feature importance from all trained classifiers
- Categorize features into 11 types (autocorrelation, statistical, stationarity, frequency, complexity, quantile, linear, count, ratio, range, symmetry, other)
- Color-coded visualizations by category
- Compare top N features across classifiers
- Category distribution analysis (pie charts + bar charts)
- Summary statistics by category

**Usage:**

```bash
# Analyze Model 1 feature importance (default: sum normalization)
python visualize_feature_importance.py --model model1

# Show top 30 features and save figures
python visualize_feature_importance.py --model model1 --top 30 --save-fig

# Use different normalization methods
python visualize_feature_importance.py --model model1 --normalize sum      # Sum to 1.0 (default)
python visualize_feature_importance.py --model model1 --normalize minmax   # Scale to 0-1
python visualize_feature_importance.py --model model1 --normalize none     # Raw values

# Analyze specific classifier only
python visualize_feature_importance.py --model model2 --classifier XGBoost --top 20

# Model 2 with figures and normalization
python visualize_feature_importance.py --model model2 --save-fig --normalize sum

# Skip plotting, only show statistics
python visualize_feature_importance.py --model model1 --no-plot
```

**Normalization Methods:**
- `sum` (default): Normalize so importances sum to 1.0 (percentages). Best for comparing relative importance across different models.
- `minmax`: Scale to 0-1 range. Preserves relative differences within each model.
- `none`: Use raw importance values from each model. Note: CatBoost uses SHAP values (larger scale), while RF/XGBoost are already normalized.

**Output:**
- Top N features comparison (horizontal bar plots, color-coded by category)
- Category distribution (pie charts showing % of total importance)
- Category importance ranking (bar charts)
- Category summary statistics (count, total, mean, max importance per category)
- High-resolution figures saved to `figures/` (150 dpi PNG)

**Supported Models:**
- `model1`: Binary classification (Stationary/Non-Stationary)
- `model2`: 5-class classification (Trend, Volatility, Stochastic, Anomaly, Structural Break)

**Feature Importance Sources:**
- RandomForest, XGBoost, CatBoost: `feature_importances_` attribute
- All sources automatically detected and loaded

## Directory Structure

```
04-postprocessing/
├── README.md                              # This documentation
├── visualize_errors_simple.py            # Batch error analysis
├── visualize_misclassified.py            # Individual time series viz
├── visualize_feature_importance.py       # Feature importance analysis
└── figures/                               # Saved visualizations (auto-created)
    ├── misclassified_15255_XGBoost.png
    ├── top_features_comparison.png
    └── category_distribution.png
```

## Quick Start

After training completes, analyze results:

```bash
cd 04-postprocessing

# 1. Analyze all errors across classifiers
python visualize_errors_simple.py --model model1 --save-fig
python visualize_errors_simple.py --model model2 --save-fig

# 2. Visualize worst misclassifications
# (Find IDs in the output of visualize_errors_simple.py)
python visualize_misclassified.py --id 15255 --model model1 --save-fig

# 3. Analyze feature importance
python visualize_feature_importance.py --model model1 --top 20 --save-fig
python visualize_feature_importance.py --model model2 --top 20 --save-fig
```

All figures will be saved to `figures/` directory.

## Installation

```bash
# Required packages
pip install numpy pandas matplotlib seaborn pyarrow
```
