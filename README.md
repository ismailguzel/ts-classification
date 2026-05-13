# Time Series Classification 

A machine learning pipeline for flat 39-class time series classification using synthetic data generated with the **betise** library. Three feature pipelines are supported: TSFresh, persistent homology (topology), and their hybrid concatenation.

---

## Quick Start

```bash
# Full dataset, TSFresh (default)
bash run.sh

# topo-test benchmark (10 classes), all three feature types
bash run.sh generation       topo-test
bash run.sh preprocessing    topo-test hybrid     # extracts both TSFresh and topology
bash run.sh training         topo-test tsfresh
bash run.sh training         topo-test topo
bash run.sh training         topo-test hybrid
```

Logs land in `logs/`. Figures in `04-postprocessing/figures/<mode>_<features>/`.

---

## Datasets

All series have fixed length = 1,000 points. Generated with the `betise` library (see [`01-data-generation/`](01-data-generation/)).

| Config | Classes | Series/class | Total | Purpose |
|--------|--------:|------------:|------:|---------|
| `full-dataset-config.json` | 39 | 1,000 | 39,000 | Full benchmark |
| `test-config.json` | 39 | 100 | 3,900 | Fast iteration |
| `topo-test-config.json` | 10 | 100 | 1,000 | Topology benchmark |

---

## Feature Pipelines

| Pipeline | Features | Source |
|----------|---------:|--------|
| **TSFresh** | 100 | `EfficientFCParameters` → MI selection |
| **Topology** | 18 | sub_H0 + sup_H0 + H1 persistence diagrams → 8 scalars each (24 raw) → MI + correlation pruning |
| **Hybrid** | 118 | TSFresh ⊕ Topology |

Topology features per diagram: 5 Carlsson coordinates + persistent entropy + L¹ and L² norms of the mean persistence landscape. See [`reports/tda-background/tda_background.pdf`](reports/tda-background/tda_background.pdf) for definitions and the [`tda_pipeline_demo.ipynb`](reports/tda-background/tda_pipeline_demo.ipynb) notebook for a worked example.

---

## Results — topo-test (10 classes, 100 series/class)

Stratified 80/20 split, seed 42. CatBoost reaches the highest mean accuracy across pipelines.

| Classifier | TSFresh (100 feat) | Topology (18 feat) | Hybrid (118 feat) |
|------------|-------------------:|-------------------:|------------------:|
| RandomForest | 93.0 % | 85.5 % | 92.5 % |
| XGBoost      | 94.5 % | 85.0 % | 94.5 % |
| **CatBoost** | **95.0 %** | **86.5 %** | **94.5 %** |
| SVM-RBF      | 89.0 % | 78.5 % | 91.0 % |

Headlines:
- **Topology with 18 features reaches 86.5 % CatBoost** — ≈ 5 × better accuracy-per-feature than TSFresh.
- **Hybrid lifts SVM most (+2 pp)**; tree-based models gain little from the merge.
- **`topo__sub_H0__carl_f3` is the #1 feature** in the hybrid CatBoost model (13.2 % importance, outranks all 100 TSFresh features).
- **Topology > TSFresh** on `trend_shift` (0.974 vs 0.919) and ties on `variance_shift`, `point_anomaly`.
- **TSFresh > Topology** on `stationary` (0.884 vs 0.651) — persistence diagrams are invariant to temporal ordering.

Full per-class F1, error analysis, and confusion patterns: [`TECHNICAL_REPORT_v2.md`](TECHNICAL_REPORT_v2.md).

---

## Project Structure

```
.
├── 01-data-generation/           # betise-based generation + JSON configs
├── 02-preprocessing/             # feature extraction (TSFresh, topology) + MI selection
├── 03-models/flat_classifier/    # RF, XGBoost, CatBoost, SVM-RBF
├── 04-postprocessing/            # confusion matrices, feature importance, error analysis
├── reports/tda-background/       # tda_background.pdf + tda_pipeline_demo.ipynb
├── data/raw/ data/features/      # parquet files
├── logs/                         # per-run logs
└── run*.sh                       # pipeline scripts
```

---

## Shell Script Reference

```bash
bash run.sh [step] [mode] [features]
```

| Argument | Options | Default |
|----------|---------|---------|
| `step`     | `all`, `generation`, `preprocessing`, `training`, `postprocessing` | `all` |
| `mode`     | `full`, `test`, `topo-test` | `full` |
| `features` | `tsfresh`, `topo`, `hybrid` | `tsfresh` |

> `preprocessing topo-test hybrid` extracts both TSFresh and topology features in a single pass; you can then train each pipeline independently without re-extracting.

---

## Documentation

| Document | Audience |
|----------|----------|
| [`TECHNICAL_REPORT_v2.md`](TECHNICAL_REPORT_v2.md) | Project technical report — results, error analysis, reproducibility |
| [`reports/tda-background/tda_background.pdf`](reports/tda-background/tda_background.pdf) | Methods / background — theory, formulas, literature review |
| [`reports/tda-background/tda_pipeline_demo.ipynb`](reports/tda-background/tda_pipeline_demo.ipynb) | TDA tutorial notebook — synthetic and real-data walk-through |
| [`01-data-generation/betise_quickstart.ipynb`](01-data-generation/betise_quickstart.ipynb) | Minimal `betise` usage examples |

---

## Requirements

- Python 3.10+
- See `requirements.txt`
