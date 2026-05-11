# Data Generation

Generates synthetic time series datasets using the **betise** library.

## Files

| File | Purpose |
|---|---|
| `generate.py` | Main generation script (reads a config JSON) |
| `full-dataset-config.json` | 39 classes × 1,000 series = 39,000 total |
| `test-config.json` | 39 classes × 100 series = 3,900 total (fast) |
| `dataset.jpeg` | Reference table of 39 target classes |

## Usage

```bash
cd 01-data-generation

# Full dataset → data/raw/dataset/dataset.parquet
python generate.py

# Test dataset → data/raw/test/test.parquet
python generate.py --config test-config.json
```

Or via shell scripts from repo root:

```bash
bash run-generation.sh        # full
bash run-generation.sh test   # test
```

## 39 Classes

| # | Class | Base | Feature overlay |
|---|---|---|---|
| 1 | stationary | ar / ma / arma / white_noise | — |
| 2 | deterministic_trend | ar | linear / quadratic / cubic / exponential trend |
| 3 | stochastic_trend | random_walk / rw_drift / ari / ima / arima | — |
| 4 | volatility | arch / garch / egarch / aparch | — |
| 5 | collective_anomaly | ar | collective_anomaly |
| 6 | contextual_anomaly | ar | contextual_anomaly |
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
