"""
Utilities for hierarchical time series classification models.

This package provides:
- metrics: Comprehensive model evaluation
- data_utils: Data loading and preprocessing
- constants: Shared constants and mappings
"""

from .metrics import ModelEvaluator
from .data_utils import (
    standardize_identifier_column,
    extract_binary_labels,
    extract_multiclass_labels,
    load_features_and_labels,
    split_train_test,
    remove_series_id_leakage
)
from .constants import (
    PRIMARY_CATEGORY_MAPPING,
    PRIMARY_CATEGORY_REVERSE,
    BINARY_CLASS_NAMES,
    PRIMARY_CLASS_NAMES,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    DEFAULT_N_JOBS
)

__all__ = [
    # Metrics
    'ModelEvaluator',
    
    # Data utilities
    'standardize_identifier_column',
    'extract_binary_labels',
    'extract_multiclass_labels',
    'load_features_and_labels',
    'split_train_test',
    'remove_series_id_leakage',
    
    # Constants
    'PRIMARY_CATEGORY_MAPPING',
    'PRIMARY_CATEGORY_REVERSE',
    'BINARY_CLASS_NAMES',
    'PRIMARY_CLASS_NAMES',
    'DEFAULT_RANDOM_STATE',
    'DEFAULT_TEST_SIZE',
    'DEFAULT_N_JOBS',
]
