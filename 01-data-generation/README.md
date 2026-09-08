# Data Generation

Generates synthetic time series datasets using the **betise** library (0.4.0).

One config per pipeline mode. Adding a mode means adding one config here and
nothing else — every path is derived from the mode name in `../modes.sh`.

## Files

`generate.py` is the entry point; it reads one of the configs below.

| Config | Study | Classes | Series |
|---|---|---:|---:|
| `shape-config.json` | A | 9 | 900 |
| `season-structure-config.json` | B1 | 4 | 400 |
| `season-anomaly-config.json` | B2 | 4 | 400 |

## Usage

```bash
cd 01-data-generation

# Study A working set -> data/raw/shape/shape.parquet
python generate.py --config shape-config.json

# Study B2 -> data/raw/season-anomaly/season-anomaly.parquet
python generate.py --config season-anomaly-config.json
```

Or via shell scripts from repo root:

```bash
bash run-generation.sh shape
bash run-generation.sh season-anomaly
```

## Seasonality is a base_series, not a feature

As of betise 0.4.0, seasonality is a property of `base_series`
(`single_seasonality`, `multiple_seasonality`, `sarma`, `sarima`) rather than an
overlay feature, and `contextual_anomaly` raises `ValueError` on any other base.
That is why `contextual_anomaly` appears only in the Study B configs.

`generate.py` passes an explicit `ALL_FEATURES_OFF` dict on every call: `load_config`
defaults are not all "off", and omitting it changes the generated series. Do not
remove it.

## Study A class taxonomy

| # | Class | Base | Feature overlay |
|---|---|---|---|
| 1 | stationary | ar / ma / arma / white_noise | — |
| 2 | deterministic_trend | ar | linear / quadratic / cubic / exponential trend |
| 3 | stochastic_trend | random_walk / rw_drift / ari / ima / arima | — |
| 4 | volatility | arch / garch / egarch / aparch | — |
| 5 | collective_anomaly | ar | collective_anomaly |
| 7 | mean_shift | ar | mean_shift |
| 8 | point_anomaly | ar | point_anomaly |
| 9 | trend_shift | ar | trend_shift |
| 10 | variance_shift | ar | variance_shift |
| 11–14 | cubic_* | ar | cubic_trend + {collective, mean_shift, point_anomaly, variance_shift} |
| 15–18 | damped_* | ar | exponential_trend(↓) + {collective, mean_shift, point_anomaly, variance_shift} |
| 19–22 | exponential_* | ar | exponential_trend(↑) + {collective, mean_shift, point_anomaly, variance_shift} |
| 23–27 | linear_* | ar | linear_trend + {collective, mean_shift, point_anomaly, trend_shift, variance_shift} |
| 28–31 | quadratic_* | ar | quadratic_trend + {collective, mean_shift, point_anomaly, variance_shift} |
| 32–35 | stochastic_* | arima | {collective, mean_shift, point_anomaly, variance_shift} |
| 36–39 | volatility_* | garch | {collective, mean_shift, point_anomaly, variance_shift} |

## Output Format

Single Apache Parquet file (long format, one row per time point):

| Column | Type | Description |
|---|---|---|
| `series_id` | int | Unique series identifier |
| `time` | int | Time index (0 … length-1) |
| `data` | float | Series value (z-normalized) |
| `is_stationary` | int | 1 = stationary, 0 = non-stationary |
| `primary_category` | str | Class name (e.g. `linear_mean_shift`) |
| `sub_category` | str | betise composite label |
| + betise metadata columns | | Trend params, anomaly indices, etc. |

## Series Length

All series: fixed **1,000 points** (`fixed_length` in config).
