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

# Primary category mapping for 5-class classification
PRIMARY_CATEGORY_MAPPING = {
    'trend': 0,
    'volatility': 1,
    'stochastic': 2,
    'anomaly': 3,
    'structural_break': 4,
}

# Reverse mapping (integer to category name)
PRIMARY_CATEGORY_REVERSE = {v: k for k, v in PRIMARY_CATEGORY_MAPPING.items()}

# ============================================================================
# CLASS NAMES
# ============================================================================

# Binary classification class names
BINARY_CLASS_NAMES = [
    'Stationary',
    'Non-Stationary'
]

# Primary category class names (5-class)
PRIMARY_CLASS_NAMES = [
    'Trend',
    'Volatility',
    'Stochastic',
    'Anomaly',
    'Structural Break'
]

# ============================================================================
# DEFAULT PARAMETERS
# ============================================================================

# Default random state for reproducibility
DEFAULT_RANDOM_STATE = 42

# Default test size for train/test split
DEFAULT_TEST_SIZE = 0.2

# Default number of parallel jobs
DEFAULT_N_JOBS = 110
