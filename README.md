# Time Series Classification with Persistent Homology

A diagnostic study of **where topological features work on time series, and what each homology dimension captures** — using synthetic data generated with the **betise** library, with TSFresh as the comparison baseline.

The question is not what accuracy is reachable. It is which *character* of series a handful of persistence descriptors separates well, which ones they cannot touch, and how that splits between H₀ and H₁. TSFresh's 100 statistical features are there to be compared against, not to be out-engineered.

The work is split into two studies, because H₁ counts loops in a delay embedding — which is a statement about *periodicity*. Keeping periodic and non-periodic classes in separate experiments is what stops the model from reading seasonality when it is supposed to be reading something else, and it is what makes H₁'s contribution interpretable.

| | **Study A — shape** | **Study B — periodicity** |
|---|---|---|
| Question | Can persistence read trends, breaks and anomalies? | Can H₁ read seasonal structure, and perturbations of it? |
| Classes | non-periodic only | seasonal only |
| Load-bearing features | `sub_H0`, `sup_H0` | `H1` |
| Role of H₁ | negative control | the protagonist |

> **Status (2026-09-08): results are being regenerated.** The previous 10-class benchmark was withdrawn after two confounds were found in it — see [Why the old results were withdrawn](#why-the-old-results-were-withdrawn). The pipeline and configs below are current; the numbers are not yet.

---

## Quick Start

Generation, feature extraction and training run on the compute host, not on a laptop.

```bash
bash run.sh all shape hybrid          # Study A: generate, extract both feature sets, train
bash run.sh all season-structure topo # Study B1: which seasonal structure?
bash run.sh all season-anomaly topo   # Study B2: what perturbs the loop?
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

| Mode | Study | Classes | Total | Purpose |
|------|-------|--------:|------:|---------|
| `shape` | A | 9 | 900 | Non-periodic benchmark; fast iteration |
| `season-structure` | B1 | 4 | 400 | `single` / `multiple` / `sarma` / `sarima` |
| `season-anomaly` | B2 | 4 | 400 | `pure` / `contextual` / `point` / `collective`, all on seasonal bases |

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

**Raw amplitude.** Nothing in the pipeline z-normalises the input series, so absolute scale flows straight into sub/superlevel persistence, which is measured in the units of the signal. Harmless in Study A, where `variance_shift` and `volatility` are *about* scale — but fatal in Study B, where per-series σ spans ~0.2 (`single_seasonality`) to ~5.7 (`sarima`).

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
| `mode`     | `shape`, `season-structure`, `season-anomaly` | `shape` |
| `features` | `tsfresh`, `topo`, `hybrid` | `tsfresh` |

> `preprocessing <mode> hybrid` extracts both TSFresh and topology features in a single pass; you can then train each pipeline independently without re-extracting.

---

## Documentation

| Document | Audience |
|----------|----------|
| [`reports/tda-background/tda_background.pdf`](reports/tda-background/tda_background.pdf) | Methods / background — theory, formulas, literature review |
| [`reports/tda-background/tda_pipeline_demo.ipynb`](reports/tda-background/tda_pipeline_demo.ipynb) | TDA tutorial notebook — synthetic and real-data walk-through |
| [`01-data-generation/betise_quickstart.ipynb`](01-data-generation/betise_quickstart.ipynb) | Minimal `betise` usage examples |
| [`TECHNICAL_REPORT_v2.md`](TECHNICAL_REPORT_v2.md) | **Superseded.** Describes the withdrawn 10-class run; retained for its error-analysis structure only |

---

## Requirements

- Python 3.11
- `requirements.txt` — pinned to the versions in use on the compute host
- `requirements-lock.txt` — full `pip freeze` of that environment
