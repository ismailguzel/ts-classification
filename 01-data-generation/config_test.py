"""
Test Dataset Configuration (15,000 samples)
============================================

Medium-sized test configuration to verify all generation code works correctly.
This generates 15,000 series with the same structure as the 150K dataset
but scaled down by 10x for reasonable testing.

Use this to test:
- All generators work correctly
- Directory structure is correct
- Output format is valid
- Pipeline runs end-to-end
- Reasonable training time

Expected runtime: ~30-60 minutes
"""

# Exact counts for 15K test distribution (150K / 10)
COUNTS_150K = {
    # ========================================================================
    # STATIONARY (7,500 total)
    # 4 base types × 1,875 each = 7,500
    # ========================================================================
    'stationary': {
        'per_base': 1875,
        'bases': ['ar', 'ma', 'arma', 'white_noise'],
        'total': 7500
    },
    
    # ========================================================================
    # DETERMINISTIC TRENDS (1,500 total)
    # 5 trend types × 2 directions × 4 bases = 40 combinations
    # 1,500 / 40 = 37.5 → 38 per combination = 1,520 (slight overcount)
    # ========================================================================
    'deterministic_trends': {
        'per_combination': 38,
        'trend_types': ['linear', 'quadratic', 'cubic', 'exponential', 'damped'],
        'directions': ['up', 'down'],
        'bases': ['ar', 'ma', 'arma', 'white_noise'],
        'total': 1520
    },
    
    # ========================================================================
    # STOCHASTIC (1,500 total)
    # 5 types × 300 each = 1,500
    # ========================================================================
    'stochastic': {
        'per_type': 300,
        'types': ['random_walk', 'random_walk_drift', 'ari', 'ima', 'arima'],
        'total': 1500
    },
    
    # ========================================================================
    # VOLATILITY (1,500 total)
    # 4 types × 375 each = 1,500
    # ========================================================================
    'volatility': {
        'per_type': 375,
        'types': ['arch', 'garch', 'egarch', 'aparch'],
        'total': 1500
    },
    
    # ========================================================================
    # POINT ANOMALIES (1,000 total, part of Anomaly group)
    # Single: 4 bases × 3 locations × 42 = 504
    # Multiple: 4 bases × 125 = 500
    # ========================================================================
    'point_anomalies': {
        'single': {
            'per_combination': 42,
            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'locations': ['beginning', 'middle', 'end'],
            'total': 504
        },
        'multiple': {
            'per_base': 125,
            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'total': 500
        }
    },
    
    # ========================================================================
    # COLLECTIVE ANOMALIES (500 total, part of Anomaly group)
    # 4 bases × 125 each = 500
    # ========================================================================
    'collective_anomalies': {
        'per_base': 125,
        'bases': ['ar', 'ma', 'arma', 'white_noise'],
        'total': 500
    },
    
    # Total Anomaly Group: 504 + 500 + 500 = 1,504
    
    # ========================================================================
    # STRUCTURAL BREAKS (1,500 total)
    # 3 break types, 500 each
    # ========================================================================
    'structural_breaks': {
        'mean_shift': {
            'per_base': 125,  # 4 × 125 = 500
            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'total': 500
        },
        'variance_shift': {
            'per_base': 125,  # 4 × 125 = 500
            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'total': 500
        },
        'trend_shift': {
            'per_combination': 62,  # 4 bases × 2 dir × 62 = 496 (close to 500)
            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'directions': ['up', 'down'],
            'total': 496
        }
    }
}

# ============================================================================
# Length Configuration
# Use same range as full dataset for consistency
# ============================================================================
LENGTH_CONFIG = {
    'short': (50, 100),      # Not used
    'medium': (300, 500),    # Not used
    'long': (1000, 5000)     # Medium range for faster generation
}

# ============================================================================
# Random Seed
# ============================================================================
RANDOM_SEED = 42

# ============================================================================
# Output Directory
# ============================================================================
OUTPUT_DIR = '../data/raw/test-15000'

# ============================================================================
# Summary
# ============================================================================
SUMMARY = """
Test Dataset (15,000 samples) Configuration:
============================================

Model 1 (Binary):
  Stationary:    7,500 (~50%)
  Unstationary:  7,520 (~50%)

Model 2 (5-class, unstationary only):
  Trend:             1,520 (~20%)
  Stochastic:        1,500 (~20%)
  Volatility:        1,500 (~20%)
  Anomaly:           1,504 (~20%)
  Structural Break:  1,496 (~20%)

All series: LONG length (1000-5000 points)
Estimated size: ~1-2 GB
Generation time: ~30-60 minutes

Purpose: Medium-size test before full 150K generation
Scaled: 10x from 1.5K test, 1/10 of 150K full dataset
"""

if __name__ == '__main__':
    print(SUMMARY)
    
    # Verify totals
    total_stationary = COUNTS_150K['stationary']['total']
    total_unstationary = (
        COUNTS_150K['deterministic_trends']['total'] +
        COUNTS_150K['stochastic']['total'] +
        COUNTS_150K['volatility']['total'] +
        COUNTS_150K['point_anomalies']['single']['total'] +
        COUNTS_150K['point_anomalies']['multiple']['total'] +
        COUNTS_150K['collective_anomalies']['total'] +
        COUNTS_150K['structural_breaks']['mean_shift']['total'] +
        COUNTS_150K['structural_breaks']['variance_shift']['total'] +
        COUNTS_150K['structural_breaks']['trend_shift']['total']
    )
    
    total = total_stationary + total_unstationary
    
    print(f"\nVerification:")
    print(f"  Stationary:   {total_stationary:,}")
    print(f"  Unstationary: {total_unstationary:,}")
    print(f"  TOTAL:        {total:,}")
    
    print(f"\n✓ Test configuration created with {total:,} series")
