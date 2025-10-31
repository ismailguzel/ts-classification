"""
Test Dataset Generation Script (1,500 samples)
===============================================

Quick test to verify all generation code works correctly.
Run this script from the 01-data-generation directory.

Usage:
    python generate_test.py

Output:
    ../data/raw/test-1500/*.parquet

Time: ~5-10 minutes
Size: ~100-200 MB
"""

import sys
import os
from pathlib import Path
import random
import numpy as np

# Import test configuration
from config_test import COUNTS_150K, LENGTH_CONFIG, RANDOM_SEED, OUTPUT_DIR

# Import ts-stationary library
from timeseries_dataset_generator import TimeSeriesGenerator
from timeseries_dataset_generator.generators import (
    # Stationary
    generate_ar_dataset,
    generate_ma_dataset,
    generate_arma_dataset,
    generate_wn_dataset,
    # Trends
    generate_linear_trend_dataset,
    generate_quadratic_trend_dataset,
    generate_cubic_trend_dataset,
    generate_exponential_trend_dataset,
    generate_damped_trend_dataset,
    # Stochastic
    generate_random_walk_dataset,
    generate_random_walk_with_drift_dataset,
    generate_ari_dataset,
    generate_ima_dataset,
    generate_arima_dataset,
    # Volatility
    generate_arch_dataset,
    generate_garch_dataset,
    generate_egarch_dataset,
    generate_aparch_dataset,
    # Anomalies
    generate_point_anomaly_dataset,
    generate_collective_anomaly_dataset,
    # Structural Breaks
    generate_mean_shift_dataset,
    generate_variance_shift_dataset,
    generate_trend_shift_dataset,
)

print("✓ ts-stationary library imported successfully")

# Set random seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Create output directory
output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)

print("="*70)
print("TEST DATASET GENERATION STARTING (1,500 samples)")
print("="*70)
print(f"Random Seed: {RANDOM_SEED}")
print(f"Output Directory: {output_path.absolute()}")
print(f"Length Range: {LENGTH_CONFIG['long']}")
print("="*70)
print()

# Helper function
def folder_path(*parts):
    """Create folder path and ensure it exists."""
    path = output_path.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)

# ============================================================================
# 1. STATIONARY (752)
# ============================================================================
print("[1/10] Generating STATIONARY series (752)...")
stationary_config = COUNTS_150K['stationary']

generators = {
    "ar": generate_ar_dataset,
    "ma": generate_ma_dataset,
    "arma": generate_arma_dataset,
    "white_noise": generate_wn_dataset
}

for base in stationary_config['bases']:
    generators[base](
        TimeSeriesGenerator,
        folder=folder_path("stationary", base, "long"),
        count=stationary_config['per_base'],
        length_range=LENGTH_CONFIG['long']
    )
print(f"  ✓ Generated {stationary_config['total']:,} stationary series\n")

# ============================================================================
# 2. DETERMINISTIC TRENDS (160)
# ============================================================================
print("[2/10] Generating DETERMINISTIC TRENDS (160)...")
trend_config = COUNTS_150K['deterministic_trends']

trend_generators = {
    'linear': generate_linear_trend_dataset,
    'quadratic': generate_quadratic_trend_dataset,
    'cubic': generate_cubic_trend_dataset,
    'exponential': generate_exponential_trend_dataset,
    'damped': generate_damped_trend_dataset
}

for trend_type in trend_config['trend_types']:
    for direction in trend_config['directions']:
        sign = 1 if direction == 'up' else -1
        for base in trend_config['bases']:
            trend_generators[trend_type](
                TimeSeriesGenerator,
                folder=folder_path(f"deterministic_trend_{trend_type}", direction, base, "long"),
                kind=base,
                count=trend_config['per_combination'],
                length_range=LENGTH_CONFIG['long'],
                sign=sign
            )
print(f"  ✓ Generated {trend_config['total']:,} trend series\n")

# ============================================================================
# 3. STOCHASTIC (150)
# ============================================================================
print("[3/10] Generating STOCHASTIC series (150)...")
stochastic_config = COUNTS_150K['stochastic']

stochastic_generators = {
    "random_walk": generate_random_walk_dataset,
    "random_walk_drift": generate_random_walk_with_drift_dataset,
    "ari": generate_ari_dataset,
    "ima": generate_ima_dataset,
    "arima": generate_arima_dataset
}

for stype in stochastic_config['types']:
    stochastic_generators[stype](
        TimeSeriesGenerator,
        folder=folder_path("stochastic", stype, "long"),
        count=stochastic_config['per_type'],
        length_range=LENGTH_CONFIG['long']
    )
print(f"  ✓ Generated {stochastic_config['total']:,} stochastic series\n")

# ============================================================================
# 4. VOLATILITY (152)
# ============================================================================
print("[4/10] Generating VOLATILITY series (152)...")
volatility_config = COUNTS_150K['volatility']

volatility_generators = {
    "arch": generate_arch_dataset,
    "garch": generate_garch_dataset,
    "egarch": generate_egarch_dataset,
    "aparch": generate_aparch_dataset
}

for vtype in volatility_config['types']:
    volatility_generators[vtype](
        TimeSeriesGenerator,
        folder=folder_path("volatility", vtype, "long"),
        count=volatility_config['per_type'],
        length_range=LENGTH_CONFIG['long']
    )
print(f"  ✓ Generated {volatility_config['total']:,} volatility series\n")

# ============================================================================
# 5. POINT ANOMALIES - SINGLE (48)
# ============================================================================
print("[5/10] Generating POINT ANOMALIES - SINGLE (48)...")
point_single_config = COUNTS_150K['point_anomalies']['single']

for base in point_single_config['bases']:
    for location in point_single_config['locations']:
        generate_point_anomaly_dataset(
            TimeSeriesGenerator,
            folder=folder_path("point_anomaly_single", base, "long"),
            kind=base,
            count=point_single_config['per_combination'],
            length_range=LENGTH_CONFIG['long'],
            anomaly_type='single',
            location=location
        )
print(f"  ✓ Generated {point_single_config['total']:,} point anomaly (single) series\n")

# ============================================================================
# 6. POINT ANOMALIES - MULTIPLE (52)
# ============================================================================
print("[6/10] Generating POINT ANOMALIES - MULTIPLE (52)...")
point_multiple_config = COUNTS_150K['point_anomalies']['multiple']

for base in point_multiple_config['bases']:
    generate_point_anomaly_dataset(
        TimeSeriesGenerator,
        folder=folder_path("point_anomaly_multiple", base, "long"),
        kind=base,
        count=point_multiple_config['per_base'],
        length_range=LENGTH_CONFIG['long'],
        anomaly_type='multiple'
    )
print(f"  ✓ Generated {point_multiple_config['total']:,} point anomaly (multiple) series\n")

# ============================================================================
# 7. COLLECTIVE ANOMALIES (52)
# ============================================================================
print("[7/10] Generating COLLECTIVE ANOMALIES (52)...")
collective_config = COUNTS_150K['collective_anomalies']

for base in collective_config['bases']:
    n = random.randint(2, 4)
    generate_collective_anomaly_dataset(
        TimeSeriesGenerator,
        folder=folder_path("multi_collective_anomaly", base, "long"),
        kind=base,
        count=collective_config['per_base'],
        num_anomalies=n,
        anomaly_type='multiple',
        length_range=LENGTH_CONFIG['long']
    )
print(f"  ✓ Generated {collective_config['total']:,} collective anomaly series\n")

# ============================================================================
# 8. STRUCTURAL BREAKS - MEAN SHIFT (52)
# ============================================================================
print("[8/10] Generating STRUCTURAL BREAKS - MEAN SHIFT (52)...")
mean_shift_config = COUNTS_150K['structural_breaks']['mean_shift']

for base in mean_shift_config['bases']:
    n = random.randint(2, 4)
    generate_mean_shift_dataset(
        TimeSeriesGenerator,
        folder=folder_path("multi_mean_shift", base, "long"),
        kind=base,
        count=mean_shift_config['per_base'],
        num_breaks=n,
        break_type='multiple',
        length_range=LENGTH_CONFIG['long']
    )
print(f"  ✓ Generated {mean_shift_config['total']:,} mean shift series\n")

# ============================================================================
# 9. STRUCTURAL BREAKS - VARIANCE SHIFT (52)
# ============================================================================
print("[9/10] Generating STRUCTURAL BREAKS - VARIANCE SHIFT (52)...")
variance_shift_config = COUNTS_150K['structural_breaks']['variance_shift']

for base in variance_shift_config['bases']:
    n = random.randint(2, 4)
    generate_variance_shift_dataset(
        TimeSeriesGenerator,
        folder=folder_path("multi_variance_shift", base, "long"),
        kind=base,
        count=variance_shift_config['per_base'],
        num_breaks=n,
        break_type='multiple',
        length_range=LENGTH_CONFIG['long']
    )
print(f"  ✓ Generated {variance_shift_config['total']:,} variance shift series\n")

# ============================================================================
# 10. STRUCTURAL BREAKS - TREND SHIFT (48)
# ============================================================================
print("[10/10] Generating STRUCTURAL BREAKS - TREND SHIFT (48)...")
trend_shift_config = COUNTS_150K['structural_breaks']['trend_shift']

change_types = ['direction_change', 'magnitude_change', 'direction_and_magnitude_change']

for base in trend_shift_config['bases']:
    for direction in trend_shift_config['directions']:
        sign = 1 if direction == 'up' else -1
        n = random.randint(2, 4)
        change_type_samples = random.choices(change_types, k=n)
        
        generate_trend_shift_dataset(
            TimeSeriesGenerator,
            folder=folder_path("multi_trend_shift", base, "long"),
            kind=base,
            count=trend_shift_config['per_combination'],
            num_breaks=n,
            change_types=change_type_samples,
            sign=sign,
            break_type='multiple',
            length_range=LENGTH_CONFIG['long']
        )
print(f"  ✓ Generated {trend_shift_config['total']:,} trend shift series\n")

# ============================================================================
# SUMMARY
# ============================================================================
# Calculate actual totals
actual_total = sum([
    stationary_config['total'],
    trend_config['total'],
    stochastic_config['total'],
    volatility_config['total'],
    point_single_config['total'],
    point_multiple_config['total'],
    collective_config['total'],
    mean_shift_config['total'],
    variance_shift_config['total'],
    trend_shift_config['total']
])

print("="*70)
print("TEST DATASET GENERATION COMPLETE!")
print("="*70)
print(f"Output directory: {output_path.absolute()}")
print(f"Total series: {actual_total:,}")
print()
print("Next steps:")
print("  1. Verify the output files were created correctly")
print("  2. Check a few sample series to ensure quality")
print("  3. If successful, run full dataset: python generate.py")
print("="*70)

