# Time Series Classification — 39 Classes

A machine learning pipeline for flat 39-class time series classification using synthetic data generated with the **betise** library. Supports three feature pipelines: TSFresh statistical features, persistent homology (topology), and a hybrid merge of both.

---

## Quick Start

```bash
# Full dataset, TSFresh (default)
bash run.sh

# Test dataset (39 classes), TSFresh
bash run.sh all test
```

### Systematic topo-test run (all three feature types)

The most efficient way to benchmark all three pipelines on the topo-test dataset:

```bash
# Step 1 — generate data once
bash run.sh generation topo-test

# Step 2 — preprocess once (hybrid covers both tsfresh + topo in one pass)
bash run.sh preprocessing topo-test hybrid

# Step 3 — train and evaluate each feature type independently
bash run.sh training topo-test topo
bash run.sh postprocessing topo-test topo

bash run.sh training topo-test tsfresh
bash run.sh postprocessing topo-test tsfresh

bash run.sh training topo-test hybrid
bash run.sh postprocessing topo-test hybrid
```

> `bash run.sh all topo-test <features>` also works but runs generation and preprocessing
> redundantly for each feature type. Use the step-by-step form above to avoid that.

All runs log to `logs/<timestamp>_<script>_<mode>_<features>.log`.  
Figures save to `04-postprocessing/figures/<mode>_<features>/`.

---

## Datasets

Generated with the **betise** library. All series have fixed length = 1,000 points.

| Config | Classes | Series/class | Total | Purpose |
|--------|---------|-------------|-------|---------|
| `full-dataset-config.json` | 39 | 1,000 | 39,000 | Full benchmark |
| `test-config.json` | 39 | 100 | 3,900 | Fast iteration |
| `topo-test-config.json` | 10 | 100 | 1,000 | Topology benchmark |

### 39-class taxonomy

| Group | Classes |
|---|---|
| Pure processes | `stationary`, `deterministic_trend`, `stochastic_trend`, `volatility` |
| Single feature | `collective_anomaly`, `contextual_anomaly`, `mean_shift`, `point_anomaly`, `trend_shift`, `variance_shift` |
| Trend × event | `cubic/damped/exponential/linear/quadratic` × `{collective, mean_shift, point_anomaly, variance_shift}` + `linear_trend_shift` |
| Process × event | `stochastic/volatility` × `{collective, mean_shift, point_anomaly, variance_shift}` |

### topo-test — 10-class topology benchmark

Pure classes 0–9 selected for topologically distinct signatures:

| Class | Topological signal |
|---|---|
| `stationary` | Many short, similar bars (high entropy) |
| `deterministic_trend` | Monotone sub_H0 bar (birth → death spans full range) |
| `stochastic_trend` | Long sub_H0 bar, unpredictable mid-range |
| `volatility` | Clustered sup_H0 bars (bursts) |
| `collective_anomaly` | Extended level shift → long sub_H0 bar |
| `contextual_anomaly` | Localised deviation — sub/sup H0 changes |
| `mean_shift` | One dominant sub_H0 bar at shift point |
| `point_anomaly` | One dominant sup_H0 bar (spike, scale_factor=5) |
| `trend_shift` | Direction change → two sub_H0 epochs |
| `variance_shift` | sup_H0 entropy change at break point |

---

## Feature Pipelines

Three pipelines are supported and can be run independently.

### TSFresh (default)

~774 raw features extracted via `EfficientFCParameters` → top **100** selected by mutual information (fitted on train split only).

```bash
bash run.sh preprocessing topo-test tsfresh
bash run.sh training      topo-test tsfresh
```

### Topology (opt-in)

Persistent homology features from the signal. Default method `sublevel+h1` produces **24 features** across three complementary diagrams.

```bash
bash run.sh preprocessing topo-test topo
bash run.sh training      topo-test topo
```

### Hybrid

Inner join of TSFresh (100) + topology features on `series_id` → **118 features** total.  
Preprocessing runs both pipelines automatically when `FEATURES=hybrid`.

```bash
bash run.sh preprocessing topo-test hybrid
bash run.sh training      topo-test hybrid
```

---

## Results — topo-test (10 classes, 100 series/class)

Train / Test split: 80 / 20 per class (stratified).

| Classifier | TSFresh (100 feat) | Topology (18 feat) | Hybrid (118 feat) |
|------------|-------------------|--------------------|-------------------|
| RandomForest | 93.5% | 84.5% | 93.0% |
| XGBoost | 94.0% | 85.0% | 94.5% |
| **CatBoost** | **95.5%** | **85.0%** | **96.5%** |
| SVM-RBF | 88.5% | 78.0% | 91.0% |

**Key observations:**
- Hybrid (CatBoost) achieves **+1.0 pp** over TSFresh alone — topology adds complementary signal
- Topology alone reaches 85% with only 18 features (24 extracted, 18 selected)
- SVM benefits most from hybrid: **+2.5 pp** (88.5% → 91.0%)
- Topology features are categorised as: Sublevel H0, Superlevel H0, Takens H0, Takens H1

---

## Topology Methods

Controlled by the `--method` flag in `topology_extraction.py` (passed via 3rd arg of `run-preprocessing.sh`).

| Method | Diagrams | Features | Description |
|--------|----------|----------|-------------|
| `sublevel+h1` | sub_H0 + sup_H0 + H1 | **24** | **Default.** No overlap — each diagram adds unique signal |
| `sublevel` | sub_H0 + sup_H0 | 16 | Sublevel/superlevel only, no Ripser |
| `takens` | H0 + H1 | 16 | Takens embedding + Vietoris-Rips (Ripser) |
| `both` | sub_H0 + sup_H0 + H0 + H1 | 32 | All diagrams |

```bash
bash run.sh preprocessing topo-test topo             # sublevel+h1 (default)
bash run-preprocessing.sh topo-test topo sublevel    # sublevel only
bash run-preprocessing.sh topo-test topo takens      # Takens + Ripser
bash run-preprocessing.sh topo-test topo both        # all diagrams
```

### Why sublevel+h1?

| Diagram | Filtration | Captures |
|---------|-----------|---------|
| **sub_H0** | `{f ≤ t}` — from below | Mean shifts, valleys, level structure |
| **sup_H0** | `{f ≥ t}` — from above | Point anomalies (spikes), peaks |
| **H1** (Takens) | Vietoris-Rips on delay embedding | Periodicity, oscillation, loops |

Takens H0 is excluded from the default: it captures similar level/clustering structure to sub/sup H0, adding redundancy without new information.

### Per-diagram features (8 per diagram)

| Feature | Formula | Captures |
|---------|---------|---------|
| `carl_f1` | Σ bᵢ(dᵢ−bᵢ) | birth-weighted persistence |
| `carl_f2` | Σ (d_max−dᵢ)(dᵢ−bᵢ) | death-proximity-weighted persistence |
| `carl_f3` | Σ bᵢ²(dᵢ−bᵢ)⁴ | high-degree birth term |
| `carl_f4` | Σ (d_max−dᵢ)²(dᵢ−bᵢ)⁴ | high-degree death term |
| `carl_f5_max` | max(dᵢ−bᵢ) | most persistent bar |
| `entropy` | −Σ(pᵢ/L)log(pᵢ/L) | diagram complexity |
| `landscape_l1` | ∫\|λ(x)\| dx | total topological signal |
| `landscape_l2` | √(∫λ(x)² dx) | dominant topological signal |

---

## Project Structure

```
.
├── 01-data-generation/
│   ├── generate.py                  # betise-based generation
│   ├── full-dataset-config.json     # 39 classes × 1,000 series
│   ├── test-config.json             # 39 classes × 100 series
│   └── topo-test-config.json        # 10 classes × 100 series (topology benchmark)
├── 02-preprocessing/
│   ├── feature_extraction.py        # TSFresh statistical features (~774 raw)
│   ├── topology_extraction.py       # Persistent homology features (24, default)
│   ├── remove_leakage_features.py   # Removes stationarity-leaking TSFresh features
│   └── feature_selection.py         # MI-based selection, fitted on train split only
├── 03-models/
│   ├── flat_classifier/
│   │   └── train.py                 # RF, XGBoost, CatBoost, SVM-RBF
│   └── utils/                       # constants, data_utils, metrics
├── 04-postprocessing/               # Confusion matrices, feature importance, error analysis
├── data/
│   ├── raw/                         # Generated parquet files
│   └── features/                    # Extracted / selected features
├── logs/                            # Per-run logs (<timestamp>_<script>_<mode>_<features>.log)
└── run*.sh                          # Pipeline automation scripts
```

---

## Shell Script Reference

### Master script

```bash
bash run.sh [step] [mode] [features]
```

| Argument | Options | Default |
|----------|---------|---------|
| `step` | `all`, `generation`, `preprocessing`, `training`, `postprocessing` | `all` |
| `mode` | `full`, `test`, `topo-test` | `full` |
| `features` | `tsfresh`, `topo`, `hybrid` | `tsfresh` |

### Individual scripts

| Script | Arguments | Notes |
|--------|-----------|-------|
| `run-generation.sh` | `[mode]` | — |
| `run-preprocessing.sh` | `[mode] [features] [topo_method]` | `hybrid` runs both TSFresh and topology |
| `run-training.sh` | `[mode] [features]` | — |
| `run-postprocessing.sh` | `[mode] [features]` | — |

Output directories:

| Features | Training output | Figures |
|----------|----------------|---------|
| `tsfresh` | `03-models/flat_classifier/output/` | `04-postprocessing/figures/<mode>_tsfresh/` |
| `topo` | `03-models/flat_classifier/output_topo/` | `04-postprocessing/figures/<mode>_topo/` |
| `hybrid` | `03-models/flat_classifier/output_hybrid/` | `04-postprocessing/figures/<mode>_hybrid/` |

---

## Requirements

- Python 3.10+
- See `requirements.txt`
