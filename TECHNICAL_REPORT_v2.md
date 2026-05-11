# Time Series Classification with TSFresh and Topological Features: Technical Report

**Author:** İsmail Güzel  
**Date:** 2026-05-11  
**Dataset:** topo-test — 1,000 Synthetic Time Series (100 series × 10 classes)  
**Architecture:** Flat 10-Class Classifier — Three Feature Pipelines (TSFresh · Topology · Hybrid)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Dataset: topo-test](#2-dataset-topo-test)
3. [Feature Extraction & Selection](#3-feature-extraction--selection)
4. [Model Training](#4-model-training)
5. [Results: Three-Pipeline Comparison](#5-results-three-pipeline-comparison)
6. [Per-Class Analysis](#6-per-class-analysis)
7. [Topological Feature Analysis](#7-topological-feature-analysis)
8. [Where TDA Adds Value](#8-where-tda-adds-value)
9. [Error Analysis](#9-error-analysis)
10. [Conclusions](#10-conclusions)
11. [Appendix: Reproducibility](#appendix-reproducibility)

---

## 1. Overview

This report evaluates a **flat 10-class time series classifier** using three independent feature pipelines:

| Pipeline | Features | Description |
|----------|----------|-------------|
| **TSFresh** | 100 | Statistical, spectral, and complexity features selected by Mutual Information |
| **Topology (TDA)** | 18 | Persistent homology features from sublevel, superlevel, and Takens filtrations |
| **Hybrid** | 118 | Concatenation of both TSFresh and topological feature sets |

The key research question is: **Can 18 topological features compete with 100 statistical features — and does combining them improve accuracy?**

### Three-Pipeline Architecture

```
Raw Time Series (length 1,000)
         │
         ├──────────────────────────────┬───────────────────────────────┐
         │                              │                               │
         ▼                              ▼                               ▼
  TSFresh Extraction            Topological Extraction          Hybrid Concat
  (774 features)               (Sublevel H0 + Superlevel H0    (TSFresh + Topo)
         │                       + Takens H1 = 24 features)           │
         ▼                              │                               │
  Leakage Removal                       ▼                               ▼
  MI Selection                   MI Selection                   StandardScaler
  (top 100)                      (top 18)                               │
         │                              │                               │
         └──────────────────────────────┴───────────────────────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │  Classifiers (per pipeline)   │
                        │  ├─ Random Forest             │
                        │  ├─ XGBoost                   │
                        │  ├─ CatBoost                  │
                        │  └─ SVM (RBF kernel)          │
                        └───────────────────────────────┘
                                        │
                                        ▼
                                10-Class Prediction
```

### The 10 Classes

These 10 classes represent the fundamental time series pattern types — the same building blocks used in the larger 39-class taxonomy:

| Class | Description |
|-------|-------------|
| `stationary` | No trend, constant variance (AR, MA, ARMA, white noise processes) |
| `deterministic_trend` | Deterministic trend (linear, quadratic, cubic, exponential shapes) |
| `stochastic_trend` | Random walk with drift (non-stationary, unit root) |
| `volatility` | Conditional heteroskedasticity (ARCH, GARCH, EGARCH, APARCH) |
| `collective_anomaly` | Sustained level deviation over a window |
| `contextual_anomaly` | Local anomaly within a specific temporal context |
| `mean_shift` | Permanent step change in the mean level |
| `point_anomaly` | Single isolated extreme value |
| `trend_shift` | Change in the slope direction (structural break) |
| `variance_shift` | Change in the variance level |

---

## 2. Dataset: topo-test

### 2.1 Generation Tool

Data is generated using the **betise** library — a parametric synthetic time series generator that supports composition of trends, seasonalities, volatility models, and anomaly injections. Configuration uses a deep-merge strategy: global defaults are overridden per class.

### 2.2 Dataset Properties

| Property | Value |
|----------|-------|
| **Total series** | 1,000 |
| **Classes** | 10 |
| **Series per class** | 100 (balanced) |
| **Series length** | 1,000 time points |
| **Random seed** | 42 |
| **Storage format** | Apache Parquet |

The dataset is **perfectly balanced** (100 series/class), removing class imbalance as a confounding factor.

### 2.3 Notable Generation Detail: point_anomaly

An important data quality fix was applied to the `point_anomaly` class. With betise's default `scale_factor=0.5`, the injected spike amplitude was only ~57% of the natural signal maximum — making the anomaly visually and statistically indistinguishable from ordinary extreme values in a stationary process.

Setting `scale_factor=5` pushes the spike to 6–9× the typical standard deviation, making point anomalies unambiguously detectable:

```json
"point_anomaly": {
  "scenarios": [
    { "base_series": "ar",  "anomaly": { "point_anomaly": { "enabled": true, "scale_factor": 5 } } },
    { "base_series": "ma",  "anomaly": { "point_anomaly": { "enabled": true, "scale_factor": 5 } } },
    { "base_series": "arma","anomaly": { "point_anomaly": { "enabled": true, "scale_factor": 5 } } }
  ]
}
```

This is why `point_anomaly` achieves **F1 = 1.00** across all three pipelines — the signal is made unambiguous by design. For classes that should be intrinsically identifiable (sharp spikes), strong generation parameters are the right choice.

---

## 3. Feature Extraction & Selection

### 3.1 TSFresh Pipeline (100 features)

**Library:** TSFresh `EfficientFCParameters`  
**Raw features extracted:** 774 per series  

TSFresh computes a broad battery of statistical, spectral, and complexity features:

| Category | Examples |
|----------|----------|
| Autocorrelation | ACF lags 1–40, partial ACF |
| Statistical | Mean, variance, skewness, kurtosis, quantiles |
| Complexity | Lempel-Ziv, sample entropy, CID |
| Frequency | FFT coefficients, spectral entropy, Fourier entropy |
| Linear trend | Slope/intercept/r-value per chunk |
| Change features | Change quantile statistics |

**Selection:** Leakage removal (ADF/KPSS statistics removed) → Mutual Information (MI) selection → **top 100 features**. MI captures non-linear dependencies, which is important for distinguishing structural patterns.

### 3.2 Topological Pipeline (18 features)

**Method:** Persistent Homology via the `sublevel+h1` method  
**Raw features extracted:** 24 per series  
**After MI selection:** 18 features  

Three complementary filtrations are computed on each time series:

| Diagram | Filtration | Captures |
|---------|-----------|---------|
| **sub_H0** | Sublevel sets (from below) | Persistence of local minima; connectivity as threshold rises |
| **sup_H0** | Superlevel sets (from above) | Persistence of local maxima; connectivity as threshold falls |
| **H1** | Takens embedding + Vietoris-Rips | Cyclic structure, loops in the delay-embedded trajectory |

Each filtration produces a **persistence diagram**. From each diagram, **8 scalar features** are extracted:

| Feature | Description |
|---------|-------------|
| `carl_f1` – `carl_f5` | Carlsson coordinates — polynomial summaries of (birth, death) pairs |
| `entropy` | Persistent entropy — complexity/spread of diagram |
| `landscape_l1` | First persistence landscape norm |
| `landscape_l2` | Second persistence landscape norm |

With 3 diagrams × 8 features = 24 total, and MI selection retaining 18, the topological representation is remarkably compact.

**Key insight: 18 features vs. 100 features.** The topological pipeline distills the time series into just 18 numbers yet achieves **85–86% accuracy** — within ~9 percentage points of the 100-feature TSFresh pipeline. This represents a **5.6× compression** of the feature space with only a moderate accuracy penalty.

### 3.3 Hybrid Pipeline (118 features)

Simple concatenation of TSFresh (100) and topological (18) features. No additional feature selection is performed before training — both feature sets are used at full length.

---

## 4. Model Training

### 4.1 Train/Test Split

| Split | Samples | Percentage | Per class |
|-------|---------|------------|-----------|
| **Train** | 800 | 80% | 80 |
| **Test** | 200 | 20% | 20 |

Stratified split by `primary_category` (fixed seed = 42).

### 4.2 Classifiers

| Classifier | Key Hyperparameters |
|-----------|---------------------|
| **Random Forest** | 200 trees, `class_weight='balanced'`, n_jobs=-1 |
| **XGBoost** | 200 estimators, `learning_rate=0.1`, `max_depth=6` |
| **CatBoost** | 200 iterations, `learning_rate=0.1`, `depth=6` |
| **SVM (RBF)** | `C=1.0`, `kernel='rbf'`, `gamma='scale'` |

All classifiers are preceded by **StandardScaler** fit on training data only. The same train/test split and scaler are applied identically across all three pipelines for fair comparison.

---

## 5. Results: Three-Pipeline Comparison

### 5.1 Accuracy Summary

| Feature Pipeline | Features | Random Forest | XGBoost | CatBoost | SVM RBF | **Mean** |
|------------------|----------|:---:|:---:|:---:|:---:|:---:|
| **TSFresh** | 100 | 93.0% | 94.5% | **95.0%** | 89.0% | 92.9% |
| **Topology** | **18** | 85.5% | 85.0% | 86.5% | 78.5% | 83.9% |
| **Hybrid** | 118 | 92.5% | 94.5% | 94.5% | **91.0%** | 93.1% |

**Best single-model result:** CatBoost + TSFresh = **95.0% accuracy**  
**Best hybrid result:** XGBoost or CatBoost + Hybrid = **94.5%**  

### 5.2 Key Observations

1. **TSFresh slightly outperforms hybrid on tree-based models.** Adding 18 topological features to 100 TSFresh features does not meaningfully improve CatBoost or XGBoost — the statistical features already capture most of the discriminating signal for this 10-class problem. The hybrid advantage shows up more in SVM (+2pp) and consistency across classifiers.

2. **Topology alone achieves 83.9% mean accuracy with only 18 features.** This is a strong result: 18 persistent homology scalars built from filtration diagrams — with no domain-specific engineering — correctly classify 8 out of 10 series on average. For a 10-class random baseline of 10%, the topological features have extracted substantial discriminative power.

3. **SVM benefits the most from topology.** SVM jumps from 89.0% (TSFresh only) to 91.0% (hybrid), while tree-based models show smaller or neutral gains. This suggests topological features provide complementary signal that linear-kernel-equivalent methods can exploit more easily alongside statistical features.

4. **CatBoost generalizes best across all pipelines.** It achieves the highest accuracy on TSFresh, second-highest on topology, and tied-highest on hybrid — while showing the smallest train-test gap, indicating superior regularization.

5. **Feature efficiency ratio.** Topology produces 88.6% of TSFresh accuracy with 18% of the features. If inference cost is a bottleneck, the topological pipeline offers a compelling accuracy-efficiency tradeoff.

---

## 6. Per-Class Analysis

### 6.1 Per-Class F1 Comparison (CatBoost, Test Set)

| Class | TSFresh F1 | Topo F1 | Hybrid F1 | Notes |
|-------|:----------:|:-------:|:---------:|-------|
| **stationary** | 0.884 | 0.651 | 0.889 | Hardest class — topology struggles; TSFresh helps |
| **deterministic_trend** | 0.974 | 0.872 | 0.974 | Topology captures trend shape but misses fine grained variants |
| **stochastic_trend** | 1.000 | 0.895 | 0.974 | TSFresh perfect; topology confused with det. trend |
| **volatility** | 0.865 | 0.750 | 0.889 | GARCH-like variance clustering is hard for persistence |
| **collective_anomaly** | 0.976 | 0.732 | 0.947 | Topology often confuses with stationary |
| **contextual_anomaly** | 1.000 | 0.974 | 1.000 | Near-perfect across all pipelines |
| **mean_shift** | 0.909 | 0.850 | 0.870 | Topology captures level shift surprisingly well |
| **point_anomaly** | **1.000** | **1.000** | **1.000** | Perfect across all pipelines — scale_factor=5 fix |
| **trend_shift** | 0.919 | 0.974 | 0.919 | **Topology outperforms TSFresh** — slope change well-captured |
| **variance_shift** | 0.974 | **0.976** | **1.000** | **Topology ties/beats TSFresh** — variance change well-captured |

### 6.2 Easiest Classes (F1 ≥ 0.97 in all pipelines)

| Class | TSFresh | Topo | Hybrid | Why |
|-------|---------|------|--------|-----|
| `point_anomaly` | 1.00 | 1.00 | 1.00 | Extreme z-score (6–9σ) is unmistakable in any feature representation |
| `contextual_anomaly` | 1.00 | 0.97 | 1.00 | Distinctive local shape with no global analog |

### 6.3 Hardest Class: stationary

`stationary` is the most difficult class across all pipelines, especially for topology (F1 = 0.651). The confusion matrix reveals that stationary series are frequently misclassified as:

- **collective_anomaly** (5 / 20 in topo CatBoost): A stationary series with a natural run of elevated values looks like a transient collective anomaly to persistence diagrams.
- **volatility** (1 / 20 in topo CatBoost): A GARCH-like clustering of variance in a stationary process resembles volatility.

TSFresh handles `stationary` much better (F1 = 0.884) because autocorrelation and spectral features can detect the absence of a trend or structural break — something that persistence diagrams, which summarize topological connectivity rather than stationarity, cannot directly measure.

---

## 7. Topological Feature Analysis

### 7.1 Feature Importance (Topology Pipeline — CatBoost)

The 18 selected topological features and their relative importances in the standalone topology model:

| Rank | Feature | Importance | Diagram | Type |
|------|---------|-----------|---------|------|
| 1 | `topo__sub_H0__carl_f3` | **25.2%** | Sublevel H0 | Carlsson coord |
| 2 | `topo__H1__landscape_l1` | 10.8% | Takens H1 | Landscape |
| 3 | `topo__H1__carl_f3` | 9.7% | Takens H1 | Carlsson coord |
| 4 | `topo__sub_H0__carl_f1` | 5.7% | Sublevel H0 | Carlsson coord |
| 5 | `topo__sup_H0__entropy` | 5.5% | Superlevel H0 | Entropy |
| 6 | `topo__H1__carl_f2` | 5.4% | Takens H1 | Carlsson coord |
| 7 | `topo__sup_H0__carl_f4` | 4.9% | Superlevel H0 | Carlsson coord |
| 8 | `topo__H1__entropy` | 4.9% | Takens H1 | Entropy |
| 9 | `topo__sub_H0__carl_f4` | 4.4% | Sublevel H0 | Carlsson coord |
| 10 | `topo__sup_H0__carl_f2` | 4.4% | Superlevel H0 | Carlsson coord |

**Per-diagram importance share:**

| Diagram | Total Importance |
|---------|-----------------|
| Sublevel H0 (sub_H0) | ~45% |
| Takens H1 | ~31% |
| Superlevel H0 (sup_H0) | ~24% |

`sub_H0__carl_f3` alone contributes 25.2% of total importance — far ahead of any other feature. This Carlsson coordinate summarizes the weighted persistence of sublevel connected components, which directly reflects how many and how persistent the local minima of the series are. This is highly informative for distinguishing trend types (which have a single dominant minimum or none) from anomaly types (which introduce secondary persistent minima).

### 7.2 Topological Features in the Hybrid Model (CatBoost)

In the hybrid model, topological features compete with 100 TSFresh features. Despite this competition, topological features remain in the top tier:

| Rank in hybrid | Feature | Importance |
|---|---|---|
| 1 | `topo__sub_H0__carl_f3` | **13.2%** — still the #1 feature overall |
| 8 | `topo__sup_H0__entropy` | 2.2% |
| 9 | `topo__sub_H0__carl_f1` | 2.2% |
| 10 | `topo__sup_H0__landscape_l1` | 2.2% |
| 17 | `topo__H1__carl_f3` | 1.6% |

`topo__sub_H0__carl_f3` is the **single most important feature in the entire hybrid model**, outranking all 100 TSFresh features. This indicates it encodes genuinely novel information that TSFresh does not capture through its extensive statistical battery.

Collectively, the ~11 topological features that survive MI selection in the hybrid contribute roughly **25–30% of total CatBoost importance** despite constituting only 15% of the feature count. Their inclusion is efficient.

---

## 8. Where TDA Adds Value

### 8.1 Classes Where Topology Is Competitive or Superior

| Class | Topo F1 | TSFresh F1 | Advantage |
|-------|---------|-----------|-----------|
| `variance_shift` | **0.976** | 0.974 | Topology matches/beats TSFresh |
| `trend_shift` | **0.974** | 0.919 | **Topology +5.5pp** |
| `contextual_anomaly` | 0.974 | 1.000 | Near-parity |
| `point_anomaly` | 1.000 | 1.000 | Perfect across the board |

**`trend_shift`** is the most striking example. A trend shift is a structural break where the slope direction changes. This creates a distinctive topological signature in the superlevel filtration (two connected components that merge at the peak of the slope change), whereas TSFresh's chunk-level linear trend features may miss the breakpoint depending on chunk alignment. The persistence diagram captures this global shape change regardless of its exact timing.

**`variance_shift`** is similarly well-captured: a change in variance creates a visible change in the density of sublevel crossings, which the Carlsson coordinates and persistence entropy encode directly.

### 8.2 Classes Where TSFresh Is Clearly Better

| Class | Topo F1 | TSFresh F1 | TSFresh advantage |
|-------|---------|-----------|-------------------|
| `stationary` | 0.651 | 0.884 | **+23.3pp** |
| `collective_anomaly` | 0.732 | 0.976 | **+24.4pp** |
| `stochastic_trend` | 0.895 | 1.000 | **+10.5pp** |
| `volatility` | 0.750 | 0.865 | **+11.5pp** |

TSFresh's advantage in these classes comes from features that have no natural topological analog:
- **Stationarity detection:** ADF-like autocorrelation structure at multiple lags distinguishes random walks from mean-reverting processes.
- **Volatility clustering:** GARCH-type behavior is detected by local variance change quantile features; this temporal clustering pattern has weak topological signature.
- **Collective anomaly duration:** The duration and exact onset of a level deviation is captured by TSFresh's `longest_strike_above_mean` and `index_mass_quantile` features — topology encodes shape but not duration as a separate scalar.

### 8.3 Summary: Complementarity

The two feature types are **structurally complementary**, not redundant:

| TSFresh strengths | Topology strengths |
|-------------------|-------------------|
| Temporal statistics (autocorrelation, GARCH) | Global shape topology (trend breaks, cycles) |
| Local duration and extent features | Scale-invariant structure |
| Stationarity proxies | Timing-independent breakpoints |
| Spectral content | Loop structure in delay-embedded space |

This is why the hybrid model shows the most consistent performance across all classifiers — it combines both information sources. The marginal accuracy gain over TSFresh alone is small at the 10-class level, but the robustness improvement (SVM: +2pp, fewer per-class extremes) is real.

---

## 9. Error Analysis

### 9.1 Misclassification Counts (Test Set, 200 samples)

| Pipeline | Model | Correct | Errors | Error Rate |
|----------|-------|---------|--------|-----------|
| **TSFresh** | CatBoost | 190 / 200 | 10 | **5.0%** |
| **TSFresh** | XGBoost | 189 / 200 | 11 | 5.5% |
| **TSFresh** | Random Forest | 186 / 200 | 14 | 7.0% |
| **TSFresh** | SVM RBF | 178 / 200 | 22 | 11.0% |
| **Topo** | CatBoost | 173 / 200 | 27 | 13.5% |
| **Topo** | Random Forest | 171 / 200 | 29 | 14.5% |
| **Topo** | XGBoost | 170 / 200 | 30 | 15.0% |
| **Topo** | SVM RBF | 157 / 200 | 43 | 21.5% |
| **Hybrid** | XGBoost | 189 / 200 | 11 | 5.5% |
| **Hybrid** | CatBoost | 189 / 200 | 11 | 5.5% |
| **Hybrid** | Random Forest | 185 / 200 | 15 | 7.5% |
| **Hybrid** | SVM RBF | 182 / 200 | 18 | 9.0% |

### 9.2 Dominant Confusion Patterns

**TSFresh and Hybrid** share the same primary confusion:
- `volatility` → `stationary` (4 / 20 in TSFresh CatBoost): Low-activity GARCH series where volatility clustering is too subtle to distinguish from stationarity.
- `trend_shift` → `mean_shift` (3 / 20 in TSFresh CatBoost): Slope changes can look like level shifts when the pre/post segments are short.

**Topology** has a distinct dominant confusion:
- `stationary` → `collective_anomaly` (5 / 20 in Topo CatBoost): A natural run of elevated values in a stationary series creates a persistent sublevel component that resembles an injected collective anomaly.
- `stationary` → `volatility` (3 / 20): A GARCH-like burst in a stationary process creates a topological signature similar to true volatility.
- `collective_anomaly` → `stationary` (3 / 20): The mirror confusion — short collective anomalies with low amplitude look topologically like stationary noise.

The fact that topology's confusions center around `stationary` while TSFresh's confusions center around `volatility` ↔ `stationary` confirms the complementarity argument: these two pipelines fail on different subsets of the test data.

### 9.3 Classes with Zero Test Errors

| Class | TSFresh | Topo | Hybrid |
|-------|---------|------|--------|
| `point_anomaly` | ✓ RF, XGB, CB | ✓ RF, XGB, CB | ✓ RF, XGB, CB |
| `contextual_anomaly` | ✓ CB | — | ✓ CB |
| `variance_shift` | ✓ CB | ✓ CB | ✓ CB, XGB |
| `stochastic_trend` | ✓ XGB, CB | — | — |
| `collective_anomaly` | ✓ XGB, CB | — | ✓ XGB, CB |

`point_anomaly` achieves zero errors across all three tree-based models in all three pipelines — a direct validation of the `scale_factor=5` generation fix.

---

## 10. Conclusions

### 10.1 Main Findings

| Finding | Detail |
|---------|--------|
| **TSFresh is the strongest single pipeline** | CatBoost 95.0% on 10 classes, 100 features |
| **TDA achieves strong accuracy with 5.6× fewer features** | 86.5% CatBoost from only 18 features |
| **Topology excels at structural breakpoints** | trend_shift F1 = 0.974 vs. TSFresh 0.919 (+5.5pp) |
| **TSFresh excels at temporal stationarity** | stationary F1 = 0.884 vs. Topo 0.651 (+23pp) |
| **Hybrid adds SVM robustness** | SVM: 89.0% → 91.0% with hybrid |
| **`sub_H0__carl_f3` is the most important hybrid feature** | Outranks all 100 TSFresh features in CatBoost |
| **point_anomaly is perfectly classified by all pipelines** | Enabled by correct scale_factor generation |

### 10.2 Feature Efficiency

```
TSFresh:  100 features → 95.0% CatBoost accuracy  →  0.950% per feature
Topology:  18 features → 86.5% CatBoost accuracy  →  4.806% per feature
```

Topology delivers **5× better accuracy per feature** than TSFresh. For applications where feature computation is expensive (real-time inference, edge deployment), the 18-feature topological pipeline offers an efficient alternative to the full statistical battery.

### 10.3 Practical Recommendations

| Scenario | Recommended Pipeline |
|----------|---------------------|
| Maximum accuracy, no compute constraint | TSFresh + CatBoost |
| Balanced accuracy + interpretability | Topology (18 features) |
| Structural break detection focus | Topology or Hybrid |
| Stationarity classification focus | TSFresh |
| Robust multi-classifier ensemble | Hybrid |

### 10.4 Extensions to the 39-Class Problem

This topo-test experiment establishes that:
1. TSFresh features at 100-feature scale achieve ~95% on 10 "pure" classes.
2. Topological features provide complementary signal for structural breaks.
3. The main challenge class is `stationary` — topologically ambiguous with anomaly and volatility classes.

For the full 39-class problem, the same pipeline structure applies. Confusion is expected to increase significantly for semantically adjacent class pairs (e.g., `linear_mean_shift` vs. `mean_shift`) where the distinguishing feature is the presence of a linear trend — well-captured by TSFresh's chunk linear trend features but only indirectly by topology.

### 10.5 Running All Three Pipelines

```bash
# topo-test dataset — run all three feature types
bash run.sh all topo-test tsfresh   # TSFresh pipeline
bash run.sh all topo-test topo      # Topology pipeline
bash run.sh all topo-test hybrid    # Hybrid pipeline

# Each step individually
bash run.sh generation topo-test
bash run.sh preprocessing topo-test tsfresh
bash run.sh training topo-test tsfresh
bash run.sh postprocessing topo-test tsfresh
```

---

## Appendix: Reproducibility

### Software Environment

```
Python:       3.10+
betise:       >=0.2.1
tsfresh:      >=0.20
giotto-tda:   >=0.6
scikit-learn: >=1.3
xgboost:      >=1.7
catboost:     >=1.2
pandas:       >=2.0
pyarrow:      >=12.0
```

### Random Seeds

| Stage | Seed |
|-------|------|
| Data generation | 42 (per class: `42 + scenario_index`) |
| Train/test split | 42 (StratifiedShuffleSplit, test_size=0.2) |
| Feature selection (MI) | 42 |
| RandomForest | 42 |
| XGBoost | 42 |
| CatBoost | 42 |

### Output Paths

| Artifact | TSFresh | Topology | Hybrid |
|----------|---------|----------|--------|
| Selected features | `data/features/topo-test/selected/primary/` | `data/features/topo-test/topological_selected/primary/` | both |
| Model outputs | `03-models/flat_classifier/output/` | `output_topo/` | `output_hybrid/` |
| Postprocessing figures | `04-postprocessing/figures/topo-test_tsfresh/` | `figures/topo-test_topo/` | `figures/topo-test_hybrid/` |

### Figure Index

| Figure | Location |
|--------|----------|
| Confusion matrix grids | `figures/{mode}_{features}/cm_grid_*.png` |
| Feature importance — top 30 | `figures/{mode}_{features}/feature_importance_top30_*.png` |
| Feature importance — categories | `figures/{mode}_{features}/feature_importance_categories_*.png` |
| Error analysis | `figures/{mode}_{features}/error_analysis_*.png` |

---

*Generated by the hierarchical-ts-classification pipeline — topo-test variant*  
*For the full 39-class architecture and taxonomy, see `README.md`.*
