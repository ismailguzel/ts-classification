"""
90K Dataset Generation Script
==============================

Uses ts-stationary library to generate balanced 90K dataset.
Run this script from the 01-data-generation directory.

Usage:
    python generate.py

Output:
    ../data/raw/unified-90k/*.parquet

Time: ~8-12 hours
Size: ~5-7 GB
"""

from pathlib import Path
import random
import numpy as np

# Import configuration
from config import COUNTS_90K, LENGTH_CONFIG, RANDOM_SEED, OUTPUT_DIR

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
print(f">>> Creating output directory: {output_path.absolute()}")
output_path.mkdir(parents=True, exist_ok=True)
print(f">>> Directory created successfully: {output_path.exists()}")

print("="*70)
print("90K DATASET GENERATION STARTING")
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
# 1. WHITE NOISE (45,000)
# ============================================================================
print("="*70)
print("[1/10] Generating WHITE NOISE series (45,000)...")
print(">>> First category starting NOW")
print("="*70)
noise_config = COUNTS_90K['white_noise']

generators = {
    "ar": generate_ar_dataset,
    "ma": generate_ma_dataset,
    "arma": generate_arma_dataset,
    "white_noise": generate_wn_dataset
}

for base in noise_config['bases']:
    print(f"  >>> Generating {base} series...")
    generators[base](
        TimeSeriesGenerator,
        folder=folder_path("stationary", base, "long"),
        count=noise_config['per_base'],
        length_range=LENGTH_CONFIG['long']
    )
    print(f"  >>> {base} completed")
print(f"  ✓ Generated {noise_config['total']:,} stationary series\n")

# ============================================================================
# 2. DETERMINISTIC TRENDS (9,000)
# ============================================================================
print("[2/10] Generating DETERMINISTIC TRENDS (9,000)...")
trend_config = COUNTS_90K['deterministic_trends']

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
# 3. STOCHASTIC (9,000)
# ============================================================================
print("[3/10] Generating STOCHASTIC series (9,000)...")
stochastic_config = COUNTS_90K['stochastic']

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
# 4. VOLATILITY (9,000)
# ============================================================================
print("[4/10] Generating VOLATILITY series (9,000)...")
volatility_config = COUNTS_90K['volatility']

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
# 5. POINT ANOMALIES - SINGLE (3,000)
# ============================================================================
print("[5/10] Generating POINT ANOMALIES - SINGLE (3,000)...")
point_single_config = COUNTS_90K['point_anomalies']['single']

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
# 6. POINT ANOMALIES - MULTIPLE (3,000)
# ============================================================================
print("[6/10] Generating POINT ANOMALIES - MULTIPLE (3,000)...")
point_multiple_config = COUNTS_90K['point_anomalies']['multiple']

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
# 7. COLLECTIVE ANOMALIES (3,000)
# ============================================================================
print("[7/10] Generating COLLECTIVE ANOMALIES (3,000)...")
collective_config = COUNTS_90K['collective_anomalies']

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
# 8. STRUCTURAL BREAKS - MEAN SHIFT (3,000)
# ============================================================================
print("[8/10] Generating STRUCTURAL BREAKS - MEAN SHIFT (3,000)...")
mean_shift_config = COUNTS_90K['structural_breaks']['mean_shift']

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
# 9. STRUCTURAL BREAKS - VARIANCE SHIFT (3,000)
# ============================================================================
print("[9/10] Generating STRUCTURAL BREAKS - VARIANCE SHIFT (3,000)...")
variance_shift_config = COUNTS_90K['structural_breaks']['variance_shift']

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
# 10. STRUCTURAL BREAKS - TREND SHIFT (3,000)
# ============================================================================
print("[10/10] Generating STRUCTURAL BREAKS - TREND SHIFT (3,000)...")
trend_shift_config = COUNTS_90K['structural_breaks']['trend_shift']

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
print("="*70)
print("DATASET GENERATION COMPLETE!")
print("="*70)
print(f"Output directory: {output_path.absolute()}")
print(f"Total series: 90,000")
print()
print("="*70)
print("NEXT STEPS")
print("="*70)
print("\nTraining Models:")
print("  • Model 1 (Binary): cd ../03-models/hierarchical/model1_binary && python train_model1.py")
print("  • Model 2 (5-class): cd ../03-models/hierarchical/model2_nonstationary && python train_model2.py")
print("\nOptional - Feature Engineering:")
print("  • TSFresh features: cd ../02-preprocessing && python extract_features.py")
print("="*70)

