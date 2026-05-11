"""
Constants used across all model training scripts.

This module centralizes:
- Category mappings (primary, sub-category)
- Class names for display
- Default parameters
"""

# ============================================================================
# CATEGORY MAPPINGS
# ============================================================================

# Primary category mapping for 39-class flat classification (dataset.jpeg order)
PRIMARY_CATEGORY_MAPPING = {
    'stationary': 0,
    'deterministic_trend': 1,
    'stochastic_trend': 2,
    'volatility': 3,
    'collective_anomaly': 4,
    'contextual_anomaly': 5,
    'mean_shift': 6,
    'point_anomaly': 7,
    'trend_shift': 8,
    'variance_shift': 9,
    'cubic_collective': 10,
    'cubic_mean_shift': 11,
    'cubic_point_anomaly': 12,
    'cubic_variance_shift': 13,
    'damped_collective': 14,
    'damped_mean_shift': 15,
    'damped_point_anomaly': 16,
    'damped_variance_shift': 17,
    'exponential_collective': 18,
    'exponential_mean_shift': 19,
    'exponential_point_anomaly': 20,
    'exponential_variance_shift': 21,
    'linear_collective': 22,
    'linear_mean_shift': 23,
    'linear_point_anomaly': 24,
    'linear_trend_shift': 25,
    'linear_variance_shift': 26,
    'quadratic_collective': 27,
    'quadratic_mean_shift': 28,
    'quadratic_point_anomaly': 29,
    'quadratic_variance_shift': 30,
    'stochastic_collective': 31,
    'stochastic_mean_shift': 32,
    'stochastic_point_anomaly': 33,
    'stochastic_variance_shift': 34,
    'volatility_collective': 35,
    'volatility_mean_shift': 36,
    'volatility_point_anomaly': 37,
    'volatility_variance_shift': 38,
}

# Reverse mapping (integer to category name)
PRIMARY_CATEGORY_REVERSE = {v: k for k, v in PRIMARY_CATEGORY_MAPPING.items()}

# ============================================================================
# CLASS NAMES
# ============================================================================

# Primary category class names (39-class), same order as mapping
PRIMARY_CLASS_NAMES = list(PRIMARY_CATEGORY_MAPPING.keys())

# ============================================================================
# DEFAULT PARAMETERS
# ============================================================================

# Default random state for reproducibility
DEFAULT_RANDOM_STATE = 42

# Default test size for train/test split
DEFAULT_TEST_SIZE = 0.2

# Default number of parallel jobs
DEFAULT_N_JOBS = -1
