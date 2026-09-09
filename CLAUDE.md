# CLAUDE.md — ts-classification

Topological data analysis (TDA) study for time series classification. Consumes the
**betise** library (developed separately at `../betise`, published on PyPI) as its data
generator; this repo is the downstream research project, not the library.

## Two machines, two roles

| | **Mac** (this one) | **munchlab1** — `iguzel@35.9.130.228` |
|---|---|---|
| Role | Authoring: code, paper/LaTeX, notebooks, git | Compute: generation, feature extraction, training |
| Source of truth for | **code and text** | **data and results** |
| Python | `/Users/iguzel/miniforge3/envs/betise-dev/bin/python` — betise packaging env; inspection only, cannot run the pipeline | `~/miniforge3/envs/ts/bin/python` — full pipeline, Python 3.11.16 |

Repo lives at `~/work/ts-classification` on munchlab1. GitHub remote is
`ismailguzel/ts-classification` (the `hierarchical-ts-classification` URL only redirects).

**Two channels, each one-way. They never cross:**

- **Code** flows Mac → GitHub → munchlab1 (`git pull`). Never edit code on munchlab1.
- **Data** stays on munchlab1. Parquet never enters git.
- **Results** (metrics JSON, importance CSV, figures) are produced on munchlab1, copied
  to the Mac, and committed **from the Mac**. They are the paper's evidence.

Do not reintroduce rsync of the code tree — an rsync copy is not a git checkout and
silently diverges. Exploratory scripts live in `~/experiments/` on munchlab1, outside
the repo; only what proves durable gets committed.

## The question

Given a time series, **is it stationary, and if not, what causes it?** — answered by a
single flat 9-class model whose labels already encode the hierarchy, with topological
features supplying the *reason* alongside the prediction.

```
input series
    ↓
one classifier (9 classes)
    ↓
non-stationary
  └─ source: deterministic trend
  └─ confidence: 0.87
  └─ why: one long sub_H0 bar, lifetime 4.2 sigma above baseline
```

No cascade. A cascade multiplies its errors (95% × 90% = 85.5%) and trains each node on
fewer samples; a flat model gives the hierarchy for free, because any label other than
`stationary` implies non-stationary, and the summed probability of the eight
non-stationary classes is a calibrated non-stationarity score that can be compared
directly against ADF / KPSS p-values.

The comparison baselines are therefore **two**: TSFresh (statistical features) and the
classical tests (ADF, KPSS, Chow). The gap the study aims at is that unit-root tests say
*whether* a series is non-stationary but not *why* — and topological features are
interpretable enough to answer that second question.

**Seasonality is out of scope**, deferred to separate work. betise 0.4.0's seasonal bases
sit at SNR 0.03–1.16 while H1 needs about 3 to resolve a loop, so seasonal classes would
contribute noise. See "Measured facts" below.

## Pipeline

```bash
bash run.sh [step] [mode] [features]
# step:     all | generation | preprocessing | training | postprocessing
# mode:     shape   (9 classes, 900 series)
# features: tsfresh | topo | hybrid
```

`modes.sh` derives every path from `$MODE`, so adding a mode means adding exactly one
file, `01-data-generation/<mode>-config.json`. `MODEL_DIR` and `FIGURES_DIR` both carry
`$MODE`, so runs cannot overwrite each other.

Two extraction switches, both defaulting **off**, both overridable per run:

- `ZSCORE=1` — per-series standardisation before the filtration. Off because
  `variance_shift` and `volatility` are genuinely *about* amplitude and sub/superlevel
  persistence is measured in the units of the signal.
- `SELF_NORM=1` — divide each diagram family's 8-vector by its own maximum. Off because
  it costs accuracy (measured 0.700 → 0.775 on `sub_H0` when switched off) and forces at
  least one feature to exactly 1.0 for every series.

`05-diagnostics/diagram_ablation.py` fits the same classifier on each diagram family
separately and prints per-class F1. Use it rather than a single lumped "topology"
accuracy — one number cannot say whether the signal came from `sub_H0`, `sup_H0` or `H1`.

## Measured facts — hand-verifiable, independently reviewed

These were measured on 2026-09-08, checked against the literature, and confirmed by an
external model review (8 of 9 claims accepted; the disputed one resolved in our favour).
They are the interpretive backbone of the paper. Do not restate the older report's
claims, which contradict several of them.

| Fact | Evidence |
|---|---|
| `sub_H0` lifetime = height of an extreme value above baseline; the peak acts as a **barrier** splitting the baseline in two | clean spike of +10 → exactly one bar of persistence 10.0 |
| A spike appears in **`sub_H0`**, not `sup_H0` — in the superlevel filtration the spike's component is the essential class, which we omit | clean spike → 0 bars in `sup_H0`; clean trough → 1 bar in `sup_H0` |
| A **clean level shift leaves no trace** in either H0 diagram; the second plateau is temporally adjacent and merges on birth | clean step → 0 bars of nonzero persistence |
| A **noisy mean shift is topologically indistinguishable from pure noise** | 338 vs 319 bars, max persistence 1.77 vs 1.79 |
| `sub_H0` reads period by counting extrema: bars = (local minima incl. endpoints) − 1 | P=20 → 50 bars, P=40 → 25 bars at n=1000. Not a general n/P law — the boundary minimum happens to cancel the essential class here |
| Sublevel persistence is **not** invariant to permuting time steps | multiset {0,1,2,3}: `[0,1,2,3]`→no bars, `[0,2,1,3]`→(1,2), `[1,3,0,2]`→(1,3) |
| H1 lifetime is **proportional to amplitude**, not period | lifetime/A = 1.759 constant for A = 0.5 … 8 |
| Period enters H1 only weakly, via τ/P; at τ/P = 0.5 the embedding collapses to a line | τ=15, P=30 → lifetime exactly 0.000 |
| H1 cannot detect periodicity below SNR ≈ 3 | 0/8 above the noise p90 at SNR 2, 7/8 at SNR 3 |
| Incommensurate frequencies give a torus: two H1 classes of near-equal persistence | 2nd/1st lifetime ratio 0.06 (P=30+P=60) vs 0.995 (P=30+P=30√2) |

### Known gaps against the canonical method

Confirmed against Perea & Harer and SW1PerS, and by the external review. They matter if
H1 is ever used seriously; they matter less for the current H0-driven study.

1. **No pointwise window normalisation.** SW1PerS mean-centres and projects each window
   vector onto the unit sphere, explicitly to make the score amplitude-blind. We
   z-normalise the whole series, which is not the same thing.
2. **τ chosen by mutual information**, which tracks noise decorrelation rather than the
   period. Perea & Harer give an explicit optimum: window `Mτ ≈ (M/(M+1))·P`.
3. **`maxdim=1` only.** The quasiperiodicity literature scores toroidality with PH1 *and*
   PH2.
4. **Only the largest bar is kept** (`carl_f5_max`). Toroidality needs the two largest H1
   classes, so the current vectorisation cannot express it.

`n_perm=500` was checked and is *not* a problem.

## Why the old results are void (2026-09-08)

Two confounds in the pre-restructure `topo-test` benchmark:

**1. The injected sine.** In betise ≤ 0.3.0, `generate_contextual_anomalies` on a
non-seasonal base injected a full-length sine of amplitude 1.5–3 × σ before placing the
anomaly. All three `contextual_anomaly` scenarios used `ar` / `arma` / `white_noise`, so
that class was the only seasonal one among ten. All 100 series had their dominant FFT
period at ≈52 or ≈91 — the only two the generator allowed at n=1000. A two-condition rule
with no learning (`45 < T < 100` and power share `> 0.40`) gives TP=100, FP=0, FN=0,
reproducing exactly the F1 = 1.000 credited to CatBoost.

**2. Raw amplitude.** Nothing z-normalised the input, so absolute scale flowed into
sub/superlevel persistence directly.

betise 0.4.0 removed the sine auto-injection and requires an explicit seasonal base for
`contextual_anomaly`, which is what surfaced the first confound.

## Open items

1. Regenerate `shape` and extract features with the new defaults (`SELF_NORM=0`).
2. Add ADF / KPSS as classical baselines — `statsmodels` is installed on munchlab1.
3. Equal-budget comparison: 24 topological features vs the **top-24** MI-selected TSFresh
   features. Comparing 24 against 100 is not a fair test and a reviewer will say so.
4. Feature attribution per prediction (SHAP or permutation importance), so the model's
   stated reason is grounded rather than asserted.

## Documentation roles — keep distinct, do not duplicate

| File | Role |
|------|------|
| `README.md` | Entry point: quick start, headline results, navigation. Keep short (~120 lines) |
| `TECHNICAL_REPORT_v2.md` | **Superseded** — the withdrawn 10-class run. Every number void |
| `reports/tda-background/tda_background.tex` → `.pdf` | Methods/background. **Contains three errors** — see "Measured facts": it claims sublevel persistence is permutation-invariant, attributes a spike to `sup_H0`, and gives a mean-shift worked example whose signature does not exist. Its results table is void |
| `reports/tda-background/tda_pipeline_demo.ipynb` | TDA tutorial |
| `01-data-generation/betise_quickstart.ipynb` | Minimal betise usage |

**All documentation and notebooks are in English.** No emoji in notebooks.

## Feature definitions — must match `02-preprocessing/topology_extraction.py`

Per persistence diagram (`sub_H0`, `sup_H0`, `H1`), 8 scalars — 24 raw features:

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

Provenance: Carlsson coordinates from Adcock et al. 2016, persistent entropy from
Chintakunta et al. 2015, landscapes from Bubenik 2015 — but Bubenik's landscape is a
*function* and we keep only two norms of it. That compression was inherited from the
background report and has never been tested; it is gap 4 above.

Selection 24 → 18 uses MI **plus correlation-based pruning**: some of the highest-MI
features (`*__carl_f5_max`, `*__landscape_l2`) are dropped as redundant with
`*__landscape_l1`. Never describe them as "low MI" — that is factually wrong.

## betise API notes (0.4.0)

- Seasonality is a property of **`base_series`** (`single_seasonality`,
  `multiple_seasonality`, `sarma`, `sarima`), **not** an overlay feature. Those names in a
  `features` dict are dead keys — 0.4.0 silently ignores them.
- `contextual_anomaly` raises `ValueError` unless the base is one of those four. Its error
  message mentions enabling "a seasonality feature", but no such path exists:
  `state["seasonal_info"]` is only set from a seasonal base (`dataset_generation.py:535`)
  and `FEATURE_ORDER` has no seasonality entry. Misleading; fixable upstream.
- `load_config` defaults are not all "off". `generate.py` passes an explicit
  `ALL_FEATURES_OFF` dict for that reason — omitting it changes the generated series.
- `generate_seasonality_from_base_series(kind, num_components, period)` has **no amplitude
  parameter**; amplitude is sampled internally (~0.139). This is why seasonal SNR cannot
  be controlled and why seasonality is deferred.

## Conventions

- Never add `Co-Authored-By: Claude` or "Generated with Claude Code" to commits or PRs
- Don't regenerate datasets or overwrite `data/` without asking
- Run the pipeline on munchlab1, never on the Mac
- Long runs go to İsmail's terminal, not a background job here; short tests are fine
- Stop and show the result at each meaningful step rather than chaining many decisions
- On munchlab1, write run output to `~/experiments/results/`, never into the repo tree.
  Results are committed from the Mac, so a copy written into a tracked path on the
  remote blocks the next `git pull` as an untracked-file collision. Pull it over with
  rsync and commit it here instead.
