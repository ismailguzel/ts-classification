# Data Generation

This directory contains scripts to generate synthetic time series datasets using the `ts-stationary` library.

## 📁 Files

| File | Purpose | Output | Runtime |
|------|---------|--------|---------|
| `config.py` | Configuration for all datasets | - | - |
| `generate_test.py` | Generate 1,450 test samples | `../data/raw/unified-test/` | ~5-10 min |
| `generate.py` | Generate 90K full dataset | `../data/raw/unified-90k/` | ~8-12 hours |

## 📊 Dataset Configurations

### 1. Test Dataset (1,450 samples)
**Config:** `COUNTS_TEST` in `config.py`

Perfect for quick testing and model development:
- **Stationary:** 752 samples (51.8%)
- **Non-Stationary:** 698 samples (48.2%)
  - Trend: 160 (22.9%)
  - Stochastic: 150 (21.5%)
  - Volatility: 148 (21.2%)
  - Anomaly: 120 (17.2%)
  - Structural Break: 120 (17.2%)

**Size:** ~100-200 MB  
**Use case:** Model development, quick iterations, debugging

### 2. Full Dataset (90K samples)
**Config:** `COUNTS_90K` in `config.py`

Production-scale balanced dataset:
- **Stationary:** 45,000 samples (50%)
- **Non-Stationary:** 45,000 samples (50%)
  - Trend: 9,000 (20%)
  - Stochastic: 9,000 (20%)
  - Volatility: 9,000 (20%)
  - Anomaly: 9,000 (20%)
  - Structural Break: 9,000 (20%)

**Size:** ~5-7 GB (Parquet compressed)  
**Use case:** Final training, production models, benchmarking

## 🚀 Usage

### Generate Test Dataset (Recommended First)

```bash
cd 01-data-generation
python generate_test.py
```

This creates `../data/raw/unified-test/` with 1,450 samples for quick testing.

### Generate Full 90K Dataset

```bash
cd 01-data-generation
python generate.py
```

This creates `../data/raw/unified-90k/` with 90,000 samples.

**⚠️ Warning:** This takes 8-12 hours and generates ~5-7 GB of data.

## 📝 Generated Data Structure

```
data/raw/
├── unified-test/              # 1,450 samples
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
└── unified-90k/               # 90K samples (same structure)
    └── ...
```

Each `.parquet` file contains:
- `series_id`: Unique identifier for each series
- `time`: Time index
- `data`: Time series values
- `is_stationary`: Boolean (True/False)
- `primary_category`: String (trend, volatility, stochastic, anomaly, structural_break)
- `sub_category`: String (specific subtype)

## 🔧 Configuration Details

### Series Length
All series use **LONG** length: 1,000 - 10,000 points

This ensures:
- Sufficient data for structural breaks
- Realistic time series characteristics
- Adequate training data for deep learning models

### Random Seed
Fixed seed (42) for reproducibility across all generations.

### Category Mapping

**Primary Categories (Model 2):**
- `trend` → Deterministic trends (linear, quadratic, cubic, exponential, damped)
- `volatility` → ARCH/GARCH patterns
- `stochastic` → Random walk, ARIMA processes
- `anomaly` → Point and collective anomalies
- `structural_break` → Mean, variance, trend shifts

## 📊 Verification

After generation, verify the dataset:

```bash
# Check test dataset
python -c "from config import COUNTS_TEST; print('Test dataset configured for:', sum([v['total'] if 'total' in v else sum([vv['total'] for vv in v.values()]) for v in COUNTS_TEST.values()]), 'samples')"

# Check full dataset
python -c "from config import COUNTS_90K; print('Full dataset configured for:', sum([v['total'] if 'total' in v else sum([vv['total'] for vv in v.values()]) for v in COUNTS_90K.values()]), 'samples')"
```

## 💡 Tips

1. **Start with test dataset** - Always test with 1,450 samples first
2. **Check disk space** - Full dataset needs ~10 GB free space
3. **Monitor progress** - Generation scripts show progress bars
4. **Reproducible** - Fixed random seed ensures identical results
5. **Metadata rich** - All series include category labels in metadata

## 🐛 Troubleshooting

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

**Last Updated:** 2025-11-01
