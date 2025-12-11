# Hierarchical Time Series Classification: A Comprehensive Study

**Technical Report**  
Date: November 19, 2025  
Dataset: 20K Synthetic Time Series

---

## Executive Summary

This report presents a comprehensive hierarchical classification system for time series stationarity detection. The system employs a two-stage architecture: first distinguishing stationary from non-stationary series (binary classification), then categorizing non-stationary patterns into five specific types (multi-class classification).                                                                              **Key Achievements:**
- **Dataset Size**: 19,980 synthetic time series samples
- **Best Model 1 Performance**: 96.55% accuracy (CatBoost) for binary classification                                                                                  - **Best Model 2 Performance**: 97.92% accuracy (XGBoost) for 5-class classification                                                                                  - **Feature Engineering**: 100 selected TSFresh features from 780+ candidates
- **Training Efficiency**: Complete pipeline execution in under 30 minutes

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset Generation](#2-dataset-generation)
3. [Feature Extraction & Selection](#3-feature-extraction--selection)
4. [Hierarchical Classification Models](#4-hierarchical-classification-models)
5. [Results & Analysis](#5-results--analysis)
6. [Post-Processing & Visualization](#6-post-processing--visualization)
7. [Conclusions & Future Work](#7-conclusions--future-work)

---

## 1. Introduction

### 1.1 Motivation

Time series stationarity is a fundamental property in statistical modeling and forecasting. Stationary series have consistent statistical properties over time, while non-stationary series exhibit trends, structural breaks, or changing volatility. Accurate classification is crucial for:                                               - **Model Selection**: Different models for stationary (ARMA) vs non-stationary (ARIMA, GARCH) data                                                                   - **Forecasting**: Stationarity assumptions affect prediction accuracy
- **Anomaly Detection**: Distinguishing true anomalies from trend/volatility changes                                                                                  - **Feature Engineering**: Time-dependent transformations for machine learning

### 1.2 Problem Statement

**Primary Task**: Binary classification of time series as stationary or non-stationary                                                                                 **Secondary Task**: Multi-class classification of non-stationary types:
1. **Trend**: Linear, quadratic, cubic, exponential, damped trends
2. **Stochastic**: Random walk, ARIMA processes
3. **Volatility**: ARCH, GARCH, EGARCH patterns
4. **Anomalies**: Point anomalies, collective anomalies
5. **Structural Breaks**: Mean shifts, variance shifts, trend changes

### 1.3 Hierarchical Approach

We employ a **two-stage hierarchical architecture**:

```
                    Input Time Series
                           |
                    [Model 1: Binary]
                    /              \
            Stationary         Non-Stationary
               (END)                 |
                            [Model 2: 5-Class]
                                     |
                    /      /    |    \      \
               Trend  Stoch  Vol  Anom  Break
```

**Benefits:**
- **Reduced Complexity**: Binary classification is simpler and more accurate
- **Focused Learning**: Model 2 only trains on non-stationary patterns
- **Interpretability**: Clear decision hierarchy
- **Modularity**: Models can be updated independently

---

## 2. Dataset Generation

### 2.1 Synthetic Data Design

**Total Samples**: 19,980 time series  
**Distribution**: 50% stationary, 50% non-stationary  
**Length Range**: 1,000-10,000 time points (long series for robust feature extraction)                                                                                **Random Seed**: 42 (reproducibility)

### 2.2 Stationary Series (9,990 samples)

Generated from classical stationary processes:

| Type | Count | Description |
|------|-------|-------------|
| AR (Autoregressive) | 2,497 | AR(p) with |p| < 1 (stable) |
| MA (Moving Average) | 2,497 | MA(q) with invertibility |
| ARMA | 2,497 | Combined AR + MA |
| White Noise | 2,499 | Gaussian i.i.d. samples |

**Parameters**: Random coefficients ensuring stationarity conditions (unit root tests pass)                                                                            ### 2.3 Non-Stationary Series (9,990 samples)

Distributed equally across five categories:

#### 2.3.1 Deterministic Trends (1,998 samples)

Systematic time-dependent patterns:
- **Linear Trends**: y(t) = α + βt (up/down)
- **Quadratic**: y(t) = α + βt + γt²
- **Cubic**: y(t) = α + βt + γt² + δt³
- **Exponential**: y(t) = α·exp(βt)
- **Damped**: y(t) = α·(1 - exp(-βt))

Each trend type combined with AR/MA/ARMA/white_noise base processes (40 combinations × ~50 samples)                                                                    #### 2.3.2 Stochastic Processes (1,998 samples)

Non-stationary random processes:
- **Random Walk**: x(t) = x(t-1) + ε(t), no reversion to mean
- **Random Walk with Drift**: x(t) = μ + x(t-1) + ε(t)
- **ARI (Autoregressive Integrated)**: ARIMA(p,d,0) with d ≥ 1
- **IMA (Integrated Moving Average)**: ARIMA(0,d,q) with d ≥ 1
- **ARIMA**: General ARIMA(p,d,q) with d ≥ 1

#### 2.3.3 Volatility Clustering (1,998 samples)

Time-varying variance (heteroskedasticity):
- **ARCH**: Autoregressive Conditional Heteroskedasticity
- **GARCH**: Generalized ARCH with persistence
- **EGARCH**: Exponential GARCH (leverage effects)
- **APARCH**: Asymmetric Power ARCH

#### 2.3.4 Anomalies (3,996 samples)

**Point Anomalies** (1,998 samples):
- Single outliers at beginning/middle/end positions
- Multiple scattered outliers (3-10 per series)

**Collective Anomalies** (1,998 samples):
- Sustained level shifts over 100-500 consecutive points
- Different magnitude and duration

#### 2.3.5 Structural Breaks (5,994 samples)

Abrupt changes in time series properties:
- **Mean Shifts** (1,998): Step changes in level
- **Variance Shifts** (1,998): Step changes in volatility
- **Trend Shifts** (1,998): Slope changes (e.g., growth rate acceleration)

### 2.4 Data Storage

**Format**: Apache Parquet (columnar, compressed)  
**Total Files**: 77 Parquet files  
**Disk Usage**: ~850 MB (raw data)

**Organization**: Data is organized by category with separate directories for:
- Stationary samples (9,990): AR, MA, ARMA, white noise
- Deterministic trends (1,960): Linear, quadratic, cubic, exponential, damped
- Stochastic processes (1,998)
- Volatility patterns (1,998)
- Point anomalies (1,998): Single and multiple
- Structural breaks (5,994): Collective anomaly, mean shift, variance shift, trend shift                                                                                ### 2.5 Label Hierarchy

Each time series has two-level labels:

**Level 1 (Binary)**:
- `0`: Stationary
- `1`: Non-Stationary

**Level 2 (5-Class, only for non-stationary)**:
- `0`: Trend (deterministic patterns)
- `1`: Stochastic (random walk, ARIMA)
- `2`: Volatility (ARCH, GARCH)
- `3`: Anomalies (point + collective)
- `4`: Structural Breaks (mean/variance/trend shifts)

---

## 3. Feature Extraction & Selection

### 3.1 TSFresh Feature Extraction

**Library**: TSFresh (Time Series Feature extraction based on scalable hypothesis tests)                                                                              **Execution Engine**: Dask distributed computing  
**Workers**: 55 parallel workers  
**Memory**: Unlimited per worker (high-memory nodes)

#### 3.1.1 Feature Set Configuration

**Selected**: MinimalFCParameters (recommended for classification)

**Initial Features**: ~780 features per time series, including:

| Category | Count | Examples |
|----------|-------|----------|
| **Autocorrelation** | ~50 | ACF lags 1-50, partial ACF |
| **Statistical** | ~80 | Mean, variance, skewness, kurtosis, percentiles |
| **Stationarity Tests** | ~15 | ADF test, KPSS test, C3 statistic |
| **Frequency Domain** | ~40 | FFT coefficients, spectral entropy |
| **Complexity** | ~30 | Approximate entropy, sample entropy, Lempel-Ziv |
| **Linear Trends** | ~20 | Trend coefficients, time-reversal asymmetry |
| **Count-based** | ~15 | Crossings, peaks, values above/below mean |
| **Ratios** | ~10 | Beyond-r-sigma ratio, large/standard deviation |
| **Quantiles** | ~100 | Q01-Q99 values |
| **Symmetry** | ~10 | Symmetry looking, time reversal |

**Total Extracted**: 780+ features × 19,980 samples = ~15.6M feature values

#### 3.1.2 Imputation & Cleaning

- **Missing Values**: Imputed with median (TSFresh default)
- **Infinite Values**: Replaced with large finite values (±1e10)
- **Constant Features**: Removed (zero variance)

**Output**: Clean feature matrix with 780+ features per sample

### 3.2 Feature Selection

**Goal**: Reduce from 780+ to 100 most relevant features

**Method**: Mutual Information (MI) based selection
- Measures dependency between features and target labels
- Non-linear relationships captured (unlike correlation)
- Works for both binary (Model 1) and multi-class (Model 2)

**Selection Process**:

1. **Separate Selection for Each Model**:
   - Model 1: Select top 100 features for binary classification
   - Model 2: Select top 100 features for 5-class classification

2. **Mutual Information Scores**:
   ```python
   from sklearn.feature_selection import mutual_info_classif
   mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
   top_features = np.argsort(mi_scores)[-100:]  # Top 100
   ```

3. **Validation**: Ensure selected features are:
   - Non-redundant (low pairwise correlation)
   - Diverse (covering multiple categories)
   - Interpretable (known TSFresh features)

**Results**:
- **Model 1 Features**: 100 features (binary classification optimized)
- **Model 2 Features**: 100 features (5-class classification optimized)

### 3.3 Feature Category Distribution

Analysis of selected features by category (Model 1 example):

| Category | Count | Percentage | Key Features |
|----------|-------|------------|--------------|
| Autocorrelation | 15 | 15% | ACF lags 1-20, partial ACF |
| Statistical | 25 | 25% | Mean, std, skewness, kurtosis, percentiles |
| Stationarity | 8 | 8% | ADF statistic, KPSS, augmented Dickey-Fuller |
| Frequency | 5 | 5% | FFT magnitudes, spectral entropy |
| Complexity | 7 | 7% | Approximate entropy, sample entropy |
| Quantiles | 20 | 20% | Q10, Q25, Q50, Q75, Q90 |
| Linear Trends | 8 | 8% | Trend coefficients, time reversal asymmetry |
| Others | 12 | 12% | Count features, ratios, symmetry |

**Key Insight**: Diverse feature representation ensures robustness across different non-stationary patterns.
