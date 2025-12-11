# Baseline Comparison: Traditional Stationarity Tests

This directory contains scripts to compute accuracy of traditional stationarity tests.

## Overview

Traditional stationarity tests:
- **ADF (Augmented Dickey-Fuller)**: Tests for unit root (H0: non-stationary)
- **KPSS (Kwiatkowski-Phillips-Schmidt-Shin)**: Tests for stationarity (H0: stationary)
- **Phillips-Perron (PP)**: Alternative unit root test with different detrending

These tests use a single test statistic and fixed threshold (typically p=0.05).

## Quick Start

### 1. Compute Baselines

```bash
# Load environment
module load apps/truba-ai/gpu-2024.0
conda activate ts-sktime

# Run baseline comparison
python compute_baselines.py \
    --data-path ../data/raw/unified-20k \
    --output-dir results/ \
    --n-jobs 100 \
    --alpha 0.05 \
    --sample-size 1000 \
    --no-dask

# Or use simple script
bash run_baseline_comparison.sh
```

**Parameters:**
- `--data-path`: Path to raw data directory
- `--output-dir`: Where to save results and figures
- `--n-jobs`: Number of parallel workers (default: 10, **recommend: 100 for TRUBA**)
- `--alpha`: Significance level for tests (default: 0.05)
- `--sample-size`: Series per pattern (default: 1000, 0=all)
- `--no-dask`: Use joblib instead of Dask (recommended)

**Runtime**: 
- Joblib (100 workers): ~10-15 minutes for 10K series
- Joblib (20 workers): ~30-40 minutes for 10K series
- Full dataset (100+ workers): ~2-3 hours

**Performance**: 
- 110 cores available on TRUBA -> Use `--n-jobs 100` for maximum speed
- Parallel processing with progress bar via joblib
- Memory-efficient execution

### 2. Output Files

```
results/
├── baseline_summary.csv              # Summary table
├── baseline_comparison.png           # Bar chart + metrics table
├── confusion_matrices.png            # Confusion matrix for each test
├── detailed_adf_test.csv             # Per-sample ADF results
├── detailed_kpss_test.csv            # Per-sample KPSS results
└── detailed_phillips-perron_test.csv # Per-sample PP results
```

### 3. Update Technical Report

After running the baseline comparison, update `TECHNICAL_REPORT.md`:

```bash
# Review results
cat results/baseline_summary.csv

# Copy figures to report directory
cp results/baseline_comparison.png ../04-postprocessing/figures/

# Update Section 5.4 with actual values
```

## Expected Results

Based on literature and typical performance:

| Method | Expected Accuracy | Limitations |
|--------|------------------|-------------|
| ADF Test | 75-85% | Fixed threshold, lag selection |
| KPSS Test | 70-82% | Trend specification |
| PP Test | 76-84% | Similar to ADF |

**Common Limitations:**
- Use only 1 test statistic
- Fixed threshold (p=0.05) regardless of data characteristics
- Sensitive to lag/trend specification
- Low power near unit root
- Fail on edge cases (volatility clustering, structural breaks)

## Dependencies

Required packages:
```bash
pip install statsmodels>=0.14.0
pip install arch>=5.0.0  # For Phillips-Perron test
pip install pandas numpy scikit-learn matplotlib seaborn tqdm joblib
```

These should already be installed in `ts-sktime` environment.

## Troubleshooting

### Issue: "arch library not available"
```bash
conda activate ts-sktime
pip install arch
```
Phillips-Perron test will be skipped if arch is not available.

### Issue: "Test file not found"
Ensure you're pointing to the correct data directory:
```bash
ls ../data/raw/unified-20k/test.parquet
```

### Issue: Slow execution
Increase parallel workers:
```bash
python compute_baselines.py --n-jobs 30 --data-path ../data/raw/unified-20k
```

### Issue: Memory error
Reduce parallel workers:
```bash
python compute_baselines.py --n-jobs 5 --data-path ../data/raw/unified-20k
```

## Technical Details

### ADF Test Implementation
```python
from statsmodels.tsa.stattools import adfuller

# H0: Unit root exists (non-stationary)
result = adfuller(series, autolag='AIC')
p_value = result[1]

if p_value < 0.05:
    prediction = "Stationary"  # Reject H0
else:
    prediction = "Non-stationary"  # Accept H0
```

### KPSS Test Implementation
```python
from statsmodels.tsa.stattools import kpss

# H0: Series is stationary
result = kpss(series, regression='c', nlags='auto')
p_value = result[1]

if p_value < 0.05:
    prediction = "Non-stationary"  # Reject H0
else:
    prediction = "Stationary"  # Accept H0
```

### Phillips-Perron Test Implementation
```python
from arch.unitroot import PhillipsPerron

# H0: Unit root exists (non-stationary)
pp = PhillipsPerron(series)
p_value = pp.pvalue

if p_value < 0.05:
    prediction = "Stationary"  # Reject H0
else:
    prediction = "Non-stationary"  # Accept H0
```

## Common Failure Cases

1. **Near-Unit-Root Processes**
   - ADF/PP: Low power near boundary (ρ ≈ 1)
   - Often misclassify borderline cases

2. **Volatility Clustering**
   - All tests: Confused by GARCH effects
   - Misinterpret variance changes as non-stationarity

3. **Structural Breaks**
   - Traditional tests: Assume stability
   - Miss trend breaks and regime changes

4. **Threshold Sensitivity**
   - p=0.05 is arbitrary choice
   - No adaptation to data characteristics

## Integration with Technical Report

After running baseline comparison:

1. **Update Section 5.4**:
   ```markdown
   | Method | Accuracy | Limitations |
   |--------|----------|-------------|
   | ADF Test (p<0.05) | XX.X% | Fixed threshold, single statistic |
   | KPSS Test | XX.X% | Sensitive to trend specification |
   | PP Test | XX.X% | Similar to ADF limitations |
   ```

2. **Add Figure**:
   - Copy `baseline_comparison.png` to `04-postprocessing/figures/`
   - Reference in report

3. **Update Summary**:
   - Replace "~70-85% (literature)" with actual computed values

## Next Steps

After baseline comparison:

1. Run `compute_baselines.py`
2. Review `results/baseline_summary.csv`
3. Copy figures to report directory
4. Update `TECHNICAL_REPORT.md` Section 5.4
5. Commit results to repository

---

**Note**: This comparison uses raw time series data with standard test parameters (α=0.05, autolag='AIC' for ADF, nlags='auto' for KPSS).

