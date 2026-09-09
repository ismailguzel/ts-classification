# Time Series Classification with Persistent Homology

Given a time series: **is it stationary, and if not, what causes it?**

A single flat 9-class model answers both at once — the labels already encode the
hierarchy, so any label other than `stationary` implies non-stationary, and the summed
probability of the eight non-stationary classes is a calibrated non-stationarity score.
Topological features supply the *reason* alongside the prediction.

The gap this aims at: unit-root tests (ADF, KPSS) say **whether** a series is
non-stationary, not **why**. Deterministic trend, stochastic trend, variance change,
level shift, a single outlier — they all return the same verdict. Persistence features
are interpretable enough to separate those causes and to say what drove the decision.

Baselines are therefore two: TSFresh (statistical features) and the classical tests.

> **Status (2026-09-09): rebuilding.** The previous 10-class benchmark was withdrawn
> after two confounds were found in it, and the background report was found to contain
> three errors about what the filtrations measure. Pipeline and configs below are
> current; results are being regenerated. See [Why the old results were withdrawn](#why-the-old-results-were-withdrawn).

---

## Quick Start

Generation, feature extraction and training run on the compute host, not on a laptop.

```bash
bash run.sh all shape hybrid          # generate, extract both feature sets, train
```

Or step by step, reusing one extraction pass across all three feature pipelines:

```bash
bash run.sh generation    shape
bash run.sh preprocessing shape hybrid    # extracts TSFresh and topology together
bash run.sh training      shape tsfresh
bash run.sh training      shape topo
bash run.sh training      shape hybrid
```

Logs land in `logs/`. Figures in `04-postprocessing/figures/<mode>_<features>/`.

---

## Datasets

All series have fixed length = 1,000 points. Generated with `betise` 0.4.0 (see [`01-data-generation/`](01-data-generation/)).

| Mode | Classes | Total | Purpose |
|------|--------:|------:|---------|
| `shape` | 9 | 900 | One stationary class, eight sources of non-stationarity |

Seasonality is out of scope and deferred to separate work: betise 0.4.0's seasonal bases
sit at SNR 0.03-1.16 while H1 needs about 3 to resolve a loop, so seasonal classes would
contribute noise rather than signal.

Modes are data-driven. Adding one means adding exactly one file, `01-data-generation/<mode>-config.json`; every path is derived from the mode name in [`modes.sh`](modes.sh).

---

## Feature Pipelines

| Pipeline | Features | Source |
|----------|---------:|--------|
| **TSFresh** | 100 | `EfficientFCParameters` → MI selection |
| **Topology** | 18 | sub_H0 + sup_H0 + H1 persistence diagrams → 8 scalars each (24 raw) → MI + correlation pruning |
| **Hybrid** | 118 | TSFresh ⊕ Topology |

Topology features per diagram: 5 Carlsson coordinates + persistent entropy + L¹ and L² norms of the mean persistence landscape. See [`reports/tda-background/tda_background.pdf`](reports/tda-background/tda_background.pdf) for definitions and [`tda_pipeline_demo.ipynb`](reports/tda-background/tda_pipeline_demo.ipynb) for a worked example.

---

## Why the old results were withdrawn

Two confounds, both the same failure mode: a nuisance property perfectly correlated with one class.

**The injected sine.** In betise ≤ 0.3.0, `generate_contextual_anomalies` on a non-seasonal base injected a full-length sine of amplitude 1.5–3 × σ before placing the anomaly. All three `contextual_anomaly` scenarios used `ar` / `arma` / `white_noise`, so every series in that class was strongly seasonal — and it was the only such class among the ten. In the generated data, all 100 of those series had their dominant FFT period at ≈52 or ≈91, the only two the generator allowed at n=1000; no other class came close. A rule with no learning at all —

```
dominant period 45 < T < 100  AND  power share > 0.40  →  contextual_anomaly
TP=100  FP=0  FN=0     F1 = 1.000
```

— reproduces exactly the F1 = 1.000 previously credited to CatBoost on that class. The model was finding the sine, not the anomaly.

**Raw amplitude.** Nothing in the pipeline z-normalised the input series, so absolute scale flowed straight into sub/superlevel persistence, which is measured in the units of the signal. That is defensible for the current 9 classes, where `variance_shift` and `volatility` are genuinely *about* scale — it was fatal for the seasonal classes, where per-series σ spanned ~0.2 to ~5.7.

**And the background report.** Independently of the data, the methods report turned out to state three things that measurement contradicts: that sublevel persistence is invariant to permuting time steps, that a point anomaly is captured by the superlevel filtration, and that a mean shift has a characteristic sublevel signature. A clean step in fact produces no bar at all, and a spike shows up in `sub_H0`. See CLAUDE.md, "Measured facts".

betise 0.4.0 removed the sine auto-injection and now requires an explicit seasonal base for `contextual_anomaly`, which is what surfaced the first confound and prompted the restructure.

---

## Project Structure

```
.
├── 01-data-generation/           # betise-based generation + one JSON config per mode
├── 02-preprocessing/             # feature extraction (TSFresh, topology) + MI selection
├── 03-models/flat_classifier/    # RF, XGBoost, CatBoost, SVM-RBF
├── 04-postprocessing/            # confusion matrices, feature importance, error analysis
├── reports/tda-background/       # tda_background.pdf + tda_pipeline_demo.ipynb
├── data/raw/ data/features/      # parquet files — never committed
├── logs/                         # per-run logs
├── modes.sh                      # mode → path mapping, sourced by every run script
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
| `mode`     | `shape` | `shape` |
| `features` | `tsfresh`, `topo`, `hybrid` | `tsfresh` |

> `preprocessing <mode> hybrid` extracts both TSFresh and topology features in a single pass; you can then train each pipeline independently without re-extracting.

---

## Documentation

| Document | Audience |
|----------|----------|
| [`reports/tda-background/tda_background.pdf`](reports/tda-background/tda_background.pdf) | Methods / background — theory, formulas, literature review. **Contains three errors** about what the filtrations measure, and its results table is void |
| [`reports/tda-background/tda_pipeline_demo.ipynb`](reports/tda-background/tda_pipeline_demo.ipynb) | TDA tutorial notebook — synthetic and real-data walk-through |
| [`01-data-generation/betise_quickstart.ipynb`](01-data-generation/betise_quickstart.ipynb) | Minimal `betise` usage examples |
| [`TECHNICAL_REPORT_v2.md`](TECHNICAL_REPORT_v2.md) | **Superseded.** Describes the withdrawn 10-class run; retained for its error-analysis structure only |

---

## Requirements

- Python 3.11
- `requirements.txt` — pinned to the versions in use on the compute host
- `requirements-lock.txt` — full `pip freeze` of that environment
