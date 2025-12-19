# Data Generation

This directory contains scripts to generate synthetic time series datasets using the `ts-stationary` library.

## Files

| File | Purpose | Output |
|------|---------|--------|
| `config.py` | Configuration for all dataset scales | - |
| `generate.py` | Generate datasets at any scale | `../data/raw/unified-{scale}/` |

## Available Dataset Scales

Strategic progression from 1.5K to 200K samples:

```bash
# View all available scales
python config.py

# Available scales:
# test, 5k, 10k, 20k, 30k, 50k, 75k, 90k, 120k, 150k, 200k
```

## Dataset Configurations

### Test Dataset (1,530 samples)
Perfect for quick testing and model development:
- **Stationary:** 765 samples (50%)
- **Non-Stationary:** 765 samples (50%)
  - Trend: 160 (20.9%)
  - Stochastic: 150 (19.6%)
  - Volatility: 148 (19.3%)
  - Anomaly: 152 (19.9%)
  - Structural Break: 144 (18.8%)

**Use case:** Model development, quick iterations, debugging

### Full Dataset Scales (5K - 200K)
Production-scale balanced datasets with consistent 50/50 split:

| Scale | Total Samples | Stationary | Non-Stationary |
|-------|---------------|------------|----------------|
| 5k    | ~5,000        | 2,520      | 2,520          |
| 10k   | ~10,000       | 4,995      | 4,995          |
| 20k   | ~20,000       | 9,990      | 9,990          |
| 30k   | ~30,000       | 14,985     | 14,985         |
| 50k   | ~50,000       | 25,020     | 25,020         |
| 75k   | ~75,000       | 37,485     | 37,485         |
| **90k** | **90,000** | **45,000** | **45,000**     |
| 120k  | ~120,000      | 59,985     | 59,985         |
| 150k  | ~150,000      | 75,015     | 75,015         |
| 200k  | ~200,000      | 99,990     | 99,990         |

All non-stationary data equally distributed across 5 classes:
- Trend: 20%
- Stochastic: 20%
- Volatility: 20%
- Anomaly: 20%
- Structural Break: 20%

**Use case:** Scaling experiments, learning curve analysis, production training

## Usage

### Generate Test Dataset (Recommended First)

```bash
cd 01-data-generation
python generate.py --scale test
```

This creates `../data/raw/unified-test/` with 1,530 samples for quick testing.

### Generate Datasets at Different Scales

```bash
# Small scale (fast, for experimentation)
python generate.py --scale 5k    # ~5 minutes
python generate.py --scale 10k   # ~10 minutes

# Medium scale
python generate.py --scale 20k
python generate.py --scale 50k

# Large scale (baseline)
python generate.py --scale 90k   # ~60-90 minutes

# Extra large scale (for scaling experiments)
python generate.py --scale 150k
python generate.py --scale 200k

### Generate with Logging

```bash
# Generate with output logging
python -u generate.py --scale 20k 2>&1 | tee generation-20k.out
```


<!-- Runtime/size estimates removed to keep usage-focused. -->

## Generated Data Structure

```
data/raw/
├── unified-test/              # 1,530 samples
│   ├── stationary/
│   │   ├── ar/
│   │   │   └── long.parquet   # Multiple series per file
│   │   ├── ma/
│   │   ├── arma/
│   │   └── white_noise/
│   ├── deterministic_trend_*/
│   ├── stochastic/
│   ├── volatility/
│   ├── point_anomaly_*/
│   ├── multi_*_anomaly/
│   ├── multi_*_shift/
│   └── ...
│
├── unified-5k/                # 5K samples (same structure)
├── unified-10k/               # 10K samples
├── unified-90k/               # 90K samples
└── unified-200k/              # 200K samples
    └── ...
```

Each `.parquet` file contains:
- `series_id`: Unique identifier for each series
- `time`: Time index
- `data`: Time series values
- `is_stationary`: Boolean (True/False)
- `primary_category`: String (trend, volatility, stochastic, anomaly, structural_break)
- `sub_category`: String (specific subtype)

## Configuration Details

### Series Length
All series use **LONG** length: 1,000 - 10,000 points

This ensures:
- Sufficient data for structural breaks
- Realistic time series characteristics
- Adequate training data for deep learning models

### Random Seed
Fixed seed (42) for reproducibility across all generations.

### Customizing Dataset Counts

View current configurations:

```bash
# List all available scales
python config.py

# View detailed breakdown for a specific scale
python config.py 5k
python config.py 90k
python config.py 200k
```

All configurations are defined in `config.py` using the `create_scaled_config()` function with a scale factor.

### Category Mapping

**Primary Categories (Model 2):**
- `trend`: Deterministic trends (linear, quadratic, cubic, exponential, damped)
- `volatility`: ARCH/GARCH patterns
- `stochastic`: Random walk, ARIMA processes
- `anomaly`: Point and collective anomalies
- `structural_break`: Mean, variance, trend shifts

## Verification

After generation, you can manually verify the dataset by checking:

- **Data structure**: Ensure all expected directories are created
- **File existence**: Check that parquet files exist for each category
- **Data loading**: Test loading a few parquet files with pandas

```bash
# Quick verification - check directory structure
ls -lh ../data/raw/unified-20k/

# Load and inspect a sample file
python -c "import pandas as pd; df = pd.read_parquet('../data/raw/unified-20k/stationary/ar/long.parquet'); print(df.head())"
```

## Tips

1. **Start with test dataset** - Always test with test scale (1.5K samples) first
2. **Check disk space** - Large datasets need ~10 GB+ free space (200K scale needs ~20 GB)
3. **Monitor progress** - Generation scripts show progress bars
4. **Reproducible** - Fixed random seed ensures identical results
5. **Metadata rich** - All series include category labels in metadata
6. **Strategic scaling** - Use smaller scales for prototyping, larger for final experiments

## Troubleshooting

**Error: ts-stationary not found**
```bash
pip install ts-stationary
```

**Out of disk space**
- Delete old datasets first
- Use test dataset for development
- Generate on machine with sufficient storage

**Generation too slow**
- Close other applications
- Use SSD if available
- Consider using test dataset only

---

**Last Updated:** 2025-11-06
