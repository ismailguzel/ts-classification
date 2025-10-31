"""
90K Balanced Dataset Configuration
====================================

Generates 90,000 series for flat vs hierarchical comparison:
- 45K Stationary
- 45K Unstationary (9K per semantic group)

All data uses LONG length (1000-10000 points) to satisfy
structural break constraints.
"""

# Exact counts for 90K balanced distribution
COUNTS_90K = {
    # ========================================================================
    # STATIONARY (45,000 total)
    # 4 base types × 11,250 each = 45,000
    # ========================================================================
    'stationary': {
        'per_base': 11250,
        'bases': ['ar', 'ma', 'arma', 'white_noise'],
        'total': 45000
    },
    
    # ========================================================================
    # DETERMINISTIC TRENDS (9,000 total)
    # 5 trend types × 2 directions × 4 bases = 40 combinations
    # 9,000 / 40 = 225 per combination
    # ========================================================================
    'deterministic_trends': {
        'per_combination': 225,
        'trend_types': ['linear', 'quadratic', 'cubic', 'exponential', 'damped'],
        'directions': ['up', 'down'],
        'bases': ['ar', 'ma', 'arma', 'white_noise'],
        'total': 9000
    },
    
    # ========================================================================
    # STOCHASTIC (9,000 total)
    # 5 types × 1,800 each = 9,000
    # ========================================================================
    'stochastic': {
        'per_type': 1800,
        'types': ['random_walk', 'random_walk_drift', 'ari', 'ima', 'arima'],
        'total': 9000
    },
    
    # ========================================================================
    # VOLATILITY (9,000 total)
    # 4 types × 2,250 each = 9,000
    # ========================================================================
    'volatility': {
        'per_type': 2250,
        'types': ['arch', 'garch', 'egarch', 'aparch'],
        'total': 9000
    },
    
    # ========================================================================
    # POINT ANOMALIES (6,000 total, part of Anomaly group)
    # Single: 4 bases × 3 locations × count = 3,000
    # Multiple: 4 bases × count = 3,000
    # ========================================================================
    'point_anomalies': {
        'single': {
            'per_combination': 250,  # 4 × 3 × 250 = 3,000
            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'locations': ['beginning', 'middle', 'end'],
            'total': 3000
        },
        'multiple': {
            'per_base': 750,  # 4 × 750 = 3,000
            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'total': 3000
        }
    },
    
    # ========================================================================
    # COLLECTIVE ANOMALIES (3,000 total, part of Anomaly group)
    # 4 bases × 750 each = 3,000
    'collective_anomalies': {
        'per_base': 750,
        'bases': ['ar', 'ma', 'arma', 'white_noise'],
        'total': 3000
    },
    'structural_breaks': {
        'mean_shift': {
            'per_base': 750,            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'total': 3000
        },
        'variance_shift': {
            'per_base': 750,            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'total': 3000
        },
        'trend_shift': {
            'per_combination': 375,            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'directions': ['up', 'down'],
            'total': 3000
        }
    }
}

# ============================================================================
# Length Configuration (ALL LONG)
# ============================================================================
LENGTH_CONFIG = {
    'short': (50, 100),      # Not used
    'medium': (300, 500),    # Not used
    'long': (1000, 10000)    # ALL data uses this
}

# ============================================================================
# Random Seed
# ============================================================================
RANDOM_SEED = 42

# ============================================================================
# Test Configuration (1,500 samples - ~1.67% of 90K)
# ============================================================================
COUNTS_TEST = {
    'stationary': {
        'per_base': 188,  # 4 × 188 = 752
        'bases': ['ar', 'ma', 'arma', 'white_noise'],
        'total': 752
    },
    'deterministic_trends': {
        'per_combination': 4,  # 5 × 2 × 4 × 4 = 160
        'trend_types': ['linear', 'quadratic', 'cubic', 'exponential', 'damped'],
        'directions': ['up', 'down'],
        'bases': ['ar', 'ma', 'arma', 'white_noise'],
        'total': 160
    },
    'stochastic': {
        'per_type': 30,  # 5 × 30 = 150
        'types': ['random_walk', 'random_walk_drift', 'ari', 'ima', 'arima'],
        'total': 150
    },
    'volatility': {
        'per_type': 38,  # 4 × 38 = 152
        'types': ['arch', 'garch', 'egarch', 'aparch'],
        'total': 152
    },
    'point_anomalies': {
        'single': {
            'per_combination': 4,  # 4 × 3 × 4 = 48
            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'locations': ['beginning', 'middle', 'end'],
            'total': 48
        },
        'multiple': {
            'per_base': 13,  # 4 × 13 = 52
            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'total': 52
        }
    },
    'collective_anomalies': {
        'per_base': 13,  # 4 × 13 = 52
        'bases': ['ar', 'ma', 'arma', 'white_noise'],
        'total': 52
    },
    'structural_breaks': {
        'mean_shift': {
            'per_base': 13,  # 4 × 13 = 52
            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'total': 52
        },
        'variance_shift': {
            'per_base': 13,  # 4 × 13 = 52
            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'total': 52
        },
        'trend_shift': {
            'per_combination': 6,  # 4 × 2 × 6 = 48
            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'directions': ['up', 'down'],
            'total': 48
        }
    }
}

# ============================================================================
# Output Directories
# ============================================================================
OUTPUT_DIR = '../data/raw/unified-90k'
OUTPUT_DIR_TEST = '../data/raw/unified-test'

# ============================================================================
# Summary
# ============================================================================
SUMMARY = """
90K Balanced Distribution Summary:
===================================

Model 1 (Binary):
  Stationary:    45,000 (50%)
  Unstationary:  45,000 (50%)

Model 2 (5-class, unstationary only):
  Trend:             9,000 (20%)
  Volatility:        9,000 (20%)
  Stochastic:        9,000 (20%)
  Anomaly:           9,000 (20%)
  Structural Break:  9,000 (20%)

Model 3 Details:
  Trend subtypes:     5 types × 1,800 = 9,000
  Volatility subtypes: 4 types × 2,250 = 9,000
  Stochastic subtypes: 5 types × 1,800 = 9,000
  Anomaly subtypes:    3 types × 3,000 = 9,000
  Structural subtypes: 3 types × 3,000 = 9,000

All series: LONG length (1000-10000 points)
Estimated size: ~7-10 GB (Parquet compressed)
Generation time: ~12-18 hours
"""

if __name__ == '__main__':
    print(SUMMARY)
    
    # Verify totals
    total_stationary = COUNTS_90K['stationary']['total']
    total_unstationary = (
        COUNTS_90K['deterministic_trends']['total'] +
        COUNTS_90K['stochastic']['total'] +
        COUNTS_90K['volatility']['total'] +
        COUNTS_90K['point_anomalies']['single']['total'] +
        COUNTS_90K['point_anomalies']['multiple']['total'] +
        COUNTS_90K['collective_anomalies']['total'] +
        COUNTS_90K['structural_breaks']['mean_shift']['total'] +
        COUNTS_90K['structural_breaks']['variance_shift']['total'] +
        COUNTS_90K['structural_breaks']['trend_shift']['total']
    )
    
    total = total_stationary + total_unstationary
    
    print(f"\nVerification:")
    print(f"  Stationary:   {total_stationary:,}")
    print(f"  Unstationary: {total_unstationary:,}")
    print(f"  TOTAL:        {total:,}")
    
    assert total == 90000, f"Total should be 90,000 but got {total:,}"
    print(f"  ✓ Verified: {total:,} series")

