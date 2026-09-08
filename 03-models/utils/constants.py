"""
Constants used across all model training scripts.

This module centralizes:
- Category mapping construction (derived from the data, not hardcoded)
- Default parameters
"""

# ============================================================================
# CATEGORY MAPPINGS
# ============================================================================

def build_category_mapping(categories) -> dict:
    """Map class names to integers, derived from the data and sorted by name.

    This used to be a hardcoded 39-class table. That broke the moment each
    pipeline mode got its own class set (9 for `shape`, 4 for each `season-*`
    study): unseen names raised, and — worse — the integer a class was given
    depended on a table nobody re-checked, so class-name lookups could silently
    label a confusion matrix wrong.

    Sorting by name keeps the assignment deterministic across runs, machines and
    train/test splits, which is all the pipeline needs.
    """
    return {name: i for i, name in enumerate(sorted({str(c) for c in categories}))}


def class_names_from_mapping(mapping: dict) -> list:
    """Class names ordered by their integer label."""
    return [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])]

# ============================================================================
# DEFAULT PARAMETERS
# ============================================================================

# Default random state for reproducibility
DEFAULT_RANDOM_STATE = 42

# Default test size for train/test split
DEFAULT_TEST_SIZE = 0.2

# Default number of parallel jobs
DEFAULT_N_JOBS = -1
