"""
150K Balanced Dataset Configuration
====================================

Generates 150,000 series for flat vs hierarchical comparison:
- 75K Stationary
- 75K Unstationary (15K per semantic group)

All data uses LONG length (1000-10000 points) to satisfy
structural break constraints.
"""

# Exact counts for 150K balanced distribution
COUNTS_150K = {
    # ========================================================================
    # STATIONARY (75,000 total)
    # 4 base types × 18,750 each = 75,000
    # ========================================================================
    'stationary': {
        'per_base': 18750,
        'bases': ['ar', 'ma', 'arma', 'white_noise'],
        'total': 75000
    },
    
    # ========================================================================
    # DETERMINISTIC TRENDS (15,000 total)
    # 5 trend types × 2 directions × 4 bases = 40 combinations
    # 15,000 / 40 = 375 per combination
    # ========================================================================
    'deterministic_trends': {
        'per_combination': 375,
        'trend_types': ['linear', 'quadratic', 'cubic', 'exponential', 'damped'],
        'directions': ['up', 'down'],
        'bases': ['ar', 'ma', 'arma', 'white_noise'],
        'total': 15000
    },
    
    # ========================================================================
    # STOCHASTIC (15,000 total)
    # 5 types × 3,000 each = 15,000
    # ========================================================================
    'stochastic': {
        'per_type': 3000,
        'types': ['random_walk', 'random_walk_drift', 'ari', 'ima', 'arima'],
        'total': 15000
    },
    
    # ========================================================================
    # VOLATILITY (15,000 total)
    # 4 types × 3,750 each = 15,000
    # ========================================================================
    'volatility': {
        'per_type': 3750,
        'types': ['arch', 'garch', 'egarch', 'aparch'],
        'total': 15000
    },
    
    # ========================================================================
    # POINT ANOMALIES (10,000 total, part of Anomaly group)
    # Single: 4 bases × 3 locations × count = 5,000
    # Multiple: 4 bases × count = 5,000
    # ========================================================================
    'point_anomalies': {
        'single': {
            'per_combination': 417,  # 4 × 3 × 417 = 5,004
            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'locations': ['beginning', 'middle', 'end'],
            'total': 5004
        },
        'multiple': {
            'per_base': 1249,  # 4 × 1,249 = 4,996
            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'total': 4996
        }
    },
    
    # ========================================================================
    # COLLECTIVE ANOMALIES (5,000 total, part of Anomaly group)
    # 4 bases × 1,250 each = 5,000
    # ========================================================================
    'collective_anomalies': {
        'per_base': 1250,
        'bases': ['ar', 'ma', 'arma', 'white_noise'],
        'total': 5000
    },
    
    # Total Anomaly Group: 5,004 + 4,996 + 5,000 = 15,000
    
    # ========================================================================
    # STRUCTURAL BREAKS (15,000 total)
    # 3 break types, distributed evenly: 5,000 each
    # ========================================================================
    'structural_breaks': {
        'mean_shift': {
            'per_base': 1250,  # 4 × 1,250 = 5,000
            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'total': 5000
        },
        'variance_shift': {
            'per_base': 1250,  # 4 × 1,250 = 5,000
            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'total': 5000
        },
        'trend_shift': {
            'per_combination': 625,  # 4 bases × 2 dir × 625 = 5,000
            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'directions': ['up', 'down'],
            'total': 5000
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
# Output Directory
# ============================================================================
OUTPUT_DIR = '../data/raw/unified-150k'

# ============================================================================
# Summary
# ============================================================================
SUMMARY = """
150K Balanced Distribution Summary:
===================================

Model 1 (Binary):
  Stationary:    75,000 (50%)
  Unstationary:  75,000 (50%)

Model 2 (5-class, unstationary only):
  Trend:             15,000 (20%)
  Volatility:        15,000 (20%)
  Stochastic:        15,000 (20%)
  Anomaly:           15,000 (20%)
  Structural Break:  15,000 (20%)

Model 3 Details:
  Trend subtypes:     5 types × 3,000 = 15,000
  Volatility subtypes: 4 types × 3,750 = 15,000
  Stochastic subtypes: 5 types × 3,000 = 15,000
  Anomaly subtypes:    3 types × 5,000 = 15,000
  Structural subtypes: 3 types × 5,000 = 15,000

All series: LONG length (1000-10000 points)
Estimated size: ~7-10 GB (Parquet compressed)
Generation time: ~12-18 hours
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
    
    assert total == 150000, f"Total should be 150,000 but got {total:,}"
    print(f"  ✓ Verified: {total:,} series")

