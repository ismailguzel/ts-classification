"""
Balanced dataset configuration for synthetic time series generation.

Includes full (90K) and test (~1.5K) distributions with category-wise
breakdowns. All series use LONG length (1000–10000).
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
# Test Configuration (1,500 samples - Perfectly Balanced)
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
        'per_type': 37,  # 4 × 37 = 148 (reduced from 152)
        'types': ['arch', 'garch', 'egarch', 'aparch'],
        'total': 148
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
            'per_base': 12,  # 4 × 12 = 48 (reduced from 52)
            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'total': 48
        },
        'variance_shift': {
            'per_base': 12,  # 4 × 12 = 48 (reduced from 52)
            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'total': 48
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
SUMMARY_90K = """
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
"""

SUMMARY_TEST = """
Test Dataset (1,500 samples) - Balanced:
========================================

Model 1 (Binary):
  Stationary:    752 (50.1%)
  Non-Stationary: 752 (49.9%)

Model 2 (5-class, non-stationary only):
  Trend:             160 (21.3%)
  Stochastic:        150 (19.9%)
  Volatility:        148 (19.7%)
  Anomaly:           152 (20.2%)  [48 single + 52 multiple + 52 collective]
  Structural Break:  144 (19.1%)  [48 mean + 48 variance + 48 trend]

Total: 1,504 samples (752 stationary + 752 non-stationary)

All series: LONG length (1000-10000 points)

Purpose: Quick test before full 90K generation
"""

SUMMARY = SUMMARY_90K  # Default to 90K summary

if __name__ == '__main__':
    import sys
    
    # Check if test mode
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        print(SUMMARY_TEST)
        config = COUNTS_TEST
        dataset_name = "Test (1.5K)"
    else:
        print(SUMMARY_90K)
        config = COUNTS_90K
        dataset_name = "90K"
    
    # Verify totals
    total_stationary = config['stationary']['total']
    total_unstationary = (
        config['deterministic_trends']['total'] +
        config['stochastic']['total'] +
        config['volatility']['total'] +
        config['point_anomalies']['single']['total'] +
        config['point_anomalies']['multiple']['total'] +
        config['collective_anomalies']['total'] +
        config['structural_breaks']['mean_shift']['total'] +
        config['structural_breaks']['variance_shift']['total'] +
        config['structural_breaks']['trend_shift']['total']
    )
    
    total = total_stationary + total_unstationary
    
    print(f"\n{dataset_name} Dataset Verification:")
    print(f"{'='*50}")
    print(f"  Stationary:     {total_stationary:>6,} ({100*total_stationary/total:.1f}%)")
    print(f"  Non-Stationary: {total_unstationary:>6,} ({100*total_unstationary/total:.1f}%)")
    print(f"  {'─'*46}")
    print(f"  TOTAL:          {total:>6,}")
    print(f"{'='*50}")
    print(f"  ✓ Balanced: {abs(total_stationary - total_unstationary) / total * 100:.2f}% difference")

