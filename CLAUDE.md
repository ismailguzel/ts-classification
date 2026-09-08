# CLAUDE.md — hierarchical-ts-classification

Topological data analysis (TDA) study for time series classification. Consumes the
**betise** library (developed separately at `../betise`, published on PyPI) as its data
generator; this repo is the downstream research project, not the library.

## Two machines, two roles

| | **Mac** (this one) | **munchlab1** — `iguzel@35.9.130.228` |
|---|---|---|
| Role | Authoring: code, paper/LaTeX, notebooks, git | Compute: generation, feature extraction, training |
| Source of truth for | **code and text** | **data and results** |
| Python | `/Users/iguzel/miniforge3/envs/betise-dev/bin/python` — betise packaging env; inspection only, cannot run the pipeline | `~/miniforge3/envs/ts/bin/python` — full pipeline, Python 3.11.16 |

**Two channels, each one-way. They never cross:**

- **Code** flows Mac → GitHub → munchlab1 (`git pull`). Never edit code on munchlab1.
  The repo is public, so the remote pulls without credentials and *cannot* push — the
  one-way rule is enforced by the auth setup, not by discipline.
- **Data** stays on munchlab1. Parquet never enters git (see `.gitignore`).
- **Results** (metrics JSON, importance CSV, figures) are produced on munchlab1, copied
  to the Mac, and committed **from the Mac**. They are the paper's evidence and must be
  versioned alongside the text. Raw data and extracted features are not.

Do not reintroduce rsync of the code tree — an rsync copy is not a git checkout and
silently diverges, leaving no way to tell which side is authoritative.

## What this study is asking

This is a **diagnostic**, not a benchmark. The question is not "what accuracy can we
reach" — it is **where do topological features work, and what does each homology
dimension actually capture**:

- On which *character* of series do persistence features separate well, and on which
  do they fail?
- What is H₀ picking up versus H₁? Seasonality is the sharpest case: H₁ counts loops
  in a delay embedding, which is a statement about periodicity, so it should carry
  Study B and contribute almost nothing to Study A.
- TSFresh is the **comparison baseline**, not a thing to optimise. The point is not to
  out-engineer it with 100 statistical features; it is to show which series a handful
  of topological descriptors handle as well or better, and which they cannot touch.

Two consequences for how work is done here:

1. **Small datasets.** Clarify the H₀/H₁ story on a few hundred series per study
   before scaling anything up. There is deliberately no large-N mode.
2. **Per-diagram results, not lumped accuracy.** Reporting one number for "topology
   (18 features)" hides the actual finding. Results must break down by diagram family
   (`sub_H0` / `sup_H0` / `H1`) and by class.

## The two studies

The paper has two sections, and the repo carries both. The split exists because
H₁ persistent homology counts loops in a delay embedding, which is a statement about
*periodicity*. Mixing periodic and non-periodic classes in one experiment lets the model
read seasonality instead of the thing being asked about — see "Why this was restructured".

**Study A — shape.** Non-periodic structure: trends, breaks, anomalies, volatility.
No class carries a seasonal component, so periodicity carries zero label information.
`sub_H0` / `sup_H0` are the load-bearing features. H₁ is expected to contribute little,
and that is the point: Study A is the negative control for H₁.

**Study B — periodicity.** Every class is seasonal, so "is it periodic" carries zero
label information and the model must read *which* periodic structure, or *what perturbs
it*. Split in two so that "which base" and "which overlay" are never conflated into one
flat label set:

- **B1 `season-structure`** — `single_seasonality` / `multiple_seasonality` / `sarma` /
  `sarima`. Which seasonal structure?
- **B2 `season-anomaly`** — `pure` / `contextual_anomaly` / `point_anomaly` /
  `collective_anomaly`, each drawing 25 series from **each** of the four seasonal bases,
  so base identity is held constant across labels and only the overlay varies.

## Pipeline

```bash
bash run.sh [step] [mode] [features]
# step:     all | generation | preprocessing | training | postprocessing
# mode:     shape | season-structure | season-anomaly
# features: tsfresh | topo | hybrid
```

`preprocessing <mode> hybrid` extracts TSFresh and topology features in one pass.

Modes are data-driven: `modes.sh` derives every path from `$MODE`, so **adding a mode
means adding exactly one file**, `01-data-generation/<mode>-config.json`. There is no
if/elif chain to update in four scripts. `MODEL_DIR` and `FIGURES_DIR` both carry
`$MODE`, so Study A and Study B runs cannot overwrite each other.

| Mode | Study | Classes | Series |
|---|---|---:|---:|
| `shape` | A | 9 | 900 |
| `season-structure` | B1 | 4 | 400 |
| `season-anomaly` | B2 | 4 | 400 |

## Why this was restructured (2026-09-08)

Two confounds were found in the pre-restructure `topo-test` results. Both were the same
failure mode — a nuisance property that happened to be perfectly correlated with one
class — and both are the reason every number in the old reports is void.

**1. The injected sine.** In betise ≤ 0.3.0, `generate_contextual_anomalies` on a
*non-seasonal* base injected a full-length sine of amplitude 1.5–3 × σ before placing the
anomaly. All three `contextual_anomaly` scenarios used `ar` / `arma` / `white_noise`, so
every series in that class was strongly seasonal — and it was the only such class among
the ten. Verified on the generated data: all 100 series had their dominant FFT period at
≈52 or ≈91, the only two periods the generator allowed at n=1000, and no other class came
close. A two-condition rule with no learning at all —

```
dominant period 45 < T < 100  AND  power share > 0.40   →  contextual_anomaly
TP=100  FP=0  FN=0     F1 = 1.000
```

— exactly reproduces the F1 = 1.000 the reports credited to CatBoost. The classifier was
detecting the sine, not the anomaly. betise 0.4.0 removed the auto-injection and now
requires an explicit seasonal base, which is what surfaced this.

**2. Raw amplitude.** Nothing in the pipeline z-normalises the input series.
`topology_extraction.py` normalises the *feature vector*; `train.py` runs `StandardScaler`
across *columns*. Sub/superlevel persistence is measured in the units of the signal, so
absolute scale flows straight into the features. In Study A that is arguably legitimate —
`variance_shift` and `volatility` are *about* scale. In Study B it is fatal: with betise
0.4.0 defaults, per-series σ ranges from ~0.2 (`single_seasonality`) to ~5.7 (`sarima`),
a 25× gap that amplitude alone would separate.

## Open decisions — settle these before generating Study B data

1. **Per-series z-normalisation for Study B is mandatory** (see confound 2). Not yet
   implemented: `topology_extraction.py` needs a per-series standardisation flag applied
   before the Takens embedding, on by default for `season-*` modes.
2. **Seasonal signal strength.** With 0.4.0 defaults the seasonal component is modest —
   ACF at the true period (30) is +0.27 for `single_seasonality`, +0.60 for `sarma`,
   +1.00 for `sarima` (seasonal unit root). Check whether `single_seasonality` produces a
   loop H₁ can actually resolve, or whether the amplitude needs raising via `params`.
3. **Takens parameters for periodic data.** The pipeline already passes `--auto-delay`
   (MI, Fraser & Swinney) and `--auto-dim` (FNN, Kennel). For a periodic signal the
   MI-selected τ should track the seasonal period, which makes τ itself an interpretable
   feature worth recording rather than discarding.

## Documentation roles — keep these distinct, do not duplicate content

| File | Role |
|------|------|
| `README.md` | Entry point: quick start, headline results, navigation. Keep it short (~120 lines) |
| `TECHNICAL_REPORT_v2.md` | **Superseded** — describes the confounded 10-class run. Retained for its error-analysis structure only; every number in it is void |
| `reports/tda-background/tda_background.tex` → `.pdf` | Methods/background: theory, formulas, literature review. Build with `make` in that dir. Its results table (p. ~737) is likewise void |
| `reports/tda-background/tda_pipeline_demo.ipynb` | TDA tutorial: Part 1 standalone synthetic, Part 2 real project data |
| `01-data-generation/betise_quickstart.ipynb` | Minimal betise usage (`load_config` + `generate_dataframe` only) |

**All documentation and notebooks are in English.** No emoji in notebooks.

## Feature definitions — must match `02-preprocessing/topology_extraction.py`

Per persistence diagram (`sub_H0`, `sup_H0`, `H1`), 8 scalars — 24 raw features total:

| Name | Formula |
|---|---|
| `carl_f1` | Σ bᵢ·pᵢ  (signed — birth can be negative) |
| `carl_f2` | Σ (d_max − dᵢ)·pᵢ |
| `carl_f3` | Σ bᵢ²·pᵢ⁴ |
| `carl_f4` | Σ (d_max − dᵢ)²·pᵢ⁴ |
| `carl_f5_max` | max pᵢ |
| `entropy` | −Σ p̂ᵢ log p̂ᵢ |
| `landscape_l1` | L¹ norm of the **mean of the top-5** persistence landscapes |
| `landscape_l2` | L² norm of that same mean vector |

Selection 24 → 18 uses MI **plus correlation-based pruning**: some of the highest-MI
features (`*__carl_f5_max`, `*__landscape_l2`) are dropped as redundant with
`*__landscape_l1`. Never describe the dropped features as "low MI" — that is factually
wrong.

Normalisation happens in **two separate blocks**: H1's 8 features are normalised among
themselves, and the 16 sub_H0+sup_H0 features among themselves. That is why a single
series can show two features equal to 1.0.

## betise API notes (0.4.0)

- Seasonality is a property of **`base_series`** (`single_seasonality`,
  `multiple_seasonality`, `sarma`, `sarima`), **not** an overlay feature. Listing those
  names in a `features` dict is a dead key — 0.4.0 silently ignores them.
- `contextual_anomaly` raises `ValueError` unless the base is one of those four. Its error
  message says "or enable a seasonality feature first", but no such path exists in 0.4.0:
  `state["seasonal_info"]` is only ever set from a seasonal base
  (`dataset_generation.py:535`), and `FEATURE_ORDER` contains no seasonality entry. The
  message is misleading and could be fixed upstream in the betise repo.
- `load_config` defaults are not all "off". `generate.py` passes an explicit
  `ALL_FEATURES_OFF` dict for exactly this reason — omitting it changes the generated
  series. Do not "simplify" it away.

## Conventions

- Never add `Co-Authored-By: Claude` or "Generated with Claude Code" to commits or PR bodies
- Don't regenerate datasets or overwrite `data/` parquet files without asking
- Run the pipeline on munchlab1, never on the Mac
- When editing a notebook programmatically, watch for `\n` inside f-strings — writing them
  as literal newlines produces `SyntaxError: unterminated f-string literal`
