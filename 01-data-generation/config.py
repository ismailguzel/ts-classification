"""
Scalable dataset configuration for synthetic time series generation.
Supports multiple scales from 1.5K (test) to 200K.
All series use LONG length (1000–10000).
"""

# ============================================================================
# Scaling Function
# ============================================================================
def create_scaled_config(scale_factor):
    """Create scaled configuration based on baseline."""
    return {
        'stationary': {
            'per_base': int(11250 * scale_factor),
            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'total': int(45000 * scale_factor)
        },
        'deterministic_trends': {
            'per_combination': int(225 * scale_factor),
            'trend_types': ['linear', 'quadratic', 'cubic', 'exponential', 'damped'],
            'directions': ['up', 'down'],
            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'total': int(9000 * scale_factor)
        },
        'stochastic': {
            'per_type': int(1800 * scale_factor),
            'types': ['random_walk', 'random_walk_drift', 'ari', 'ima', 'arima'],
            'total': int(9000 * scale_factor)
        },
        'volatility': {
            'per_type': int(2250 * scale_factor),
            'types': ['arch', 'garch', 'egarch', 'aparch'],
            'total': int(9000 * scale_factor)
        },
        'point_anomalies': {
            'single': {
                'per_combination': int(250 * scale_factor),
                'bases': ['ar', 'ma', 'arma', 'white_noise'],
                'locations': ['beginning', 'middle', 'end'],
                'total': int(3000 * scale_factor)
            },
            'multiple': {
                'per_base': int(750 * scale_factor),
                'bases': ['ar', 'ma', 'arma', 'white_noise'],
                'total': int(3000 * scale_factor)
            }
        },
        'collective_anomalies': {
            'per_base': int(750 * scale_factor),
            'bases': ['ar', 'ma', 'arma', 'white_noise'],
            'total': int(3000 * scale_factor)
        },
        'structural_breaks': {
            'mean_shift': {
                'per_base': int(750 * scale_factor),
                'bases': ['ar', 'ma', 'arma', 'white_noise'],
                'total': int(3000 * scale_factor)
            },
            'variance_shift': {
                'per_base': int(750 * scale_factor),
                'bases': ['ar', 'ma', 'arma', 'white_noise'],
                'total': int(3000 * scale_factor)
            },
            'trend_shift': {
                'per_combination': int(375 * scale_factor),
                'bases': ['ar', 'ma', 'arma', 'white_noise'],
                'directions': ['up', 'down'],
                'total': int(3000 * scale_factor)
            }
        }
    }

# ============================================================================
# Scale Presets (Strategic Progression: 1.5K -> 200K)
# ============================================================================
SCALE_CONFIGS = {
    'test':  create_scaled_config(0.017),    # ~1,500 samples
    '5k':    create_scaled_config(0.056),    # ~5,000 samples
    '10k':   create_scaled_config(0.111),    # ~10,000 samples
    '20k':   create_scaled_config(0.222),    # ~20,000 samples
    '30k':   create_scaled_config(0.333),    # ~30,000 samples
    '50k':   create_scaled_config(0.556),    # ~50,000 samples
    '75k':   create_scaled_config(0.833),    # ~75,000 samples
    '90k':   create_scaled_config(1.0),      # 90,000 samples (baseline)
    '120k':  create_scaled_config(1.333),    # ~120,000 samples
    '150k':  create_scaled_config(1.667),    # ~150,000 samples
    '200k':  create_scaled_config(2.222)     # ~200,000 samples
}

# Backward compatibility
COUNTS_TEST = SCALE_CONFIGS['test']

# ============================================================================
# Global Settings
# ============================================================================
LENGTH_CONFIG = {
    'long': (1000, 2000)    # All datasets use this
}

RANDOM_SEED = 42

def get_output_dir(scale):
    """Get output directory for given scale."""
    return f'../data/raw/unified-{scale}'

# ============================================================================
# Helper Functions
# ============================================================================
def print_scale_summary(scale_name, config):
    """Print summary for a given scale."""
    total_stat = config['stationary']['total']
    
    # Calculate 5-class breakdown
    trend_total = config['deterministic_trends']['total']
    stochastic_total = config['stochastic']['total']
    volatility_total = config['volatility']['total']
    anomaly_total = (config['point_anomalies']['single']['total'] + 
                     config['point_anomalies']['multiple']['total'] + 
                     config['collective_anomalies']['total'])
    structural_total = (config['structural_breaks']['mean_shift']['total'] + 
                        config['structural_breaks']['variance_shift']['total'] + 
                        config['structural_breaks']['trend_shift']['total'])
    
    total_nonstat = trend_total + stochastic_total + volatility_total + anomaly_total + structural_total
    total = total_stat + total_nonstat
    
    print(f"\n{scale_name.upper()} Dataset Configuration:")
    print("="*60)
    print(f"  Total:          {total:>7,} samples")
    print(f"  Stationary:     {total_stat:>7,} ({100*total_stat/total:.1f}%)")
    print(f"  Non-Stationary: {total_nonstat:>7,} ({100*total_nonstat/total:.1f}%)")
    print("="*60)
    print("\n  Non-Stationary Breakdown (5-Class):")
    print(f"    Trend:            {trend_total:>7,} ({100*trend_total/total_nonstat:.1f}%)")
    print(f"    Stochastic:       {stochastic_total:>7,} ({100*stochastic_total/total_nonstat:.1f}%)")
    print(f"    Volatility:       {volatility_total:>7,} ({100*volatility_total/total_nonstat:.1f}%)")
    print(f"    Anomaly:          {anomaly_total:>7,} ({100*anomaly_total/total_nonstat:.1f}%)")
    print(f"    Structural Break: {structural_total:>7,} ({100*structural_total/total_nonstat:.1f}%)")
    print("="*60)

# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    import sys
    
    print("\nAvailable Dataset Scales:")
    print("="*60)
    print(f"{'Scale':<8} {'Total':>10} {'Stationary':>12} {'Non-Stat':>12}")
    print("-"*60)
    
    for scale_name in ['test', '5k', '10k', '20k', '30k', '50k', '75k', '90k', '120k', '150k', '200k']:
        config = SCALE_CONFIGS[scale_name]
        total_stat = config['stationary']['total']
        total_nonstat = sum([
            config['deterministic_trends']['total'],
            config['stochastic']['total'],
            config['volatility']['total'],
            config['point_anomalies']['single']['total'],
            config['point_anomalies']['multiple']['total'],
            config['collective_anomalies']['total'],
            config['structural_breaks']['mean_shift']['total'],
            config['structural_breaks']['variance_shift']['total'],
            config['structural_breaks']['trend_shift']['total']
        ])
        total = total_stat + total_nonstat
        print(f"{scale_name:<8} {total:>10,} {total_stat:>12,} {total_nonstat:>12,}")
    
    print("="*60)
    
    if len(sys.argv) > 1:
        scale = sys.argv[1]
        if scale in SCALE_CONFIGS:
            print_scale_summary(scale, SCALE_CONFIGS[scale])
        else:
            print(f"\nError: Unknown scale '{scale}'")
            print(f"Available: {list(SCALE_CONFIGS.keys())}")
