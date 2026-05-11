"""
Data utilities for consistent pipeline operations.

This module provides standardized functions for:
- Identifier column handling (series_id vs id)
- Label extraction (binary, multi-class)
- Data loading with validation
- Train/test splitting

All training scripts should use these functions to ensure consistency.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from .constants import (
    PRIMARY_CATEGORY_MAPPING,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
)


def standardize_identifier_column(df, target_name='series_id', set_as_index=False):
    """
    Ensure identifier column is named consistently throughout pipeline.
    
    Args:
        df: DataFrame with 'id' or 'series_id' column
        target_name: Target column name (default: 'series_id')
        set_as_index: If True, set as index after renaming
    
    Returns:
        DataFrame with standardized identifier column
    
    Raises:
        ValueError: If no identifier column found
    """
    df = df.copy()
    
    # Already has target name
    if target_name in df.columns:
        if set_as_index:
            return df.set_index(target_name)
        return df
    
    # Has 'id', rename to target
    if 'id' in df.columns:
        df = df.rename(columns={'id': target_name})
        if set_as_index:
            return df.set_index(target_name)
        return df
    
    # Check if it's already in index
    if df.index.name == target_name:
        if set_as_index:
            return df
        return df.reset_index()
    
    if df.index.name == 'id':
        df.index.name = target_name
        if set_as_index:
            return df
        return df.reset_index()

    # Has 'index' column (common artifact of reset_index)
    if 'index' in df.columns:
        # Verify it looks like an ID (integer)
        if pd.api.types.is_integer_dtype(df['index']):
            df = df.rename(columns={'index': target_name})
            if set_as_index:
                return df.set_index(target_name)
            return df
    
    raise ValueError(
        f"No identifier column found. Expected '{target_name}' or 'id'. "
        f"Available columns: {list(df.columns)}, Index name: {df.index.name}"
    )


def extract_binary_labels(labels_df, positive_class='non-stationary'):
    """
    Extract binary labels from is_stationary column.
    
    Args:
        labels_df: DataFrame with 'is_stationary' column
        positive_class: Which class is positive (default: 'non-stationary')
    
    Returns:
        numpy array with 0 (stationary) and 1 (non-stationary)
    
    Raises:
        ValueError: If 'is_stationary' column not found
    """
    if 'is_stationary' not in labels_df.columns:
        raise ValueError(
            f"Missing 'is_stationary' column in labels. "
            f"Available columns: {list(labels_df.columns)}"
        )
    
    # is_stationary can be bool or int (1=stationary, 0=non-stationary)
    # Convert to consistent format: 0=stationary, 1=non-stationary
    is_stat = labels_df['is_stationary'].astype(bool)
    
    if positive_class == 'non-stationary':
        # 0 = stationary, 1 = non-stationary
        return (~is_stat).astype(int).values
    else:
        # 0 = non-stationary, 1 = stationary
        return is_stat.astype(int).values


def extract_multiclass_labels(labels_df, category_mapping=None, 
                               category_col='primary_category'):
    """
    Extract multi-class labels from category column.
    
    Args:
        labels_df: DataFrame with category column
        category_mapping: Dict mapping category names to integers
                         (default: PRIMARY_CATEGORY_MAPPING)
        category_col: Name of category column (default: 'primary_category')
    
    Returns:
        numpy array with integer labels
    
    Raises:
        ValueError: If category column not found or unknown categories exist
    """
    if category_col not in labels_df.columns:
        raise ValueError(
            f"Missing '{category_col}' column in labels. "
            f"Available columns: {list(labels_df.columns)}"
        )
    
    if category_mapping is None:
        category_mapping = PRIMARY_CATEGORY_MAPPING
    
    # Map categories to integers
    labels = labels_df[category_col].map(category_mapping)
    
    # Check for unknown categories
    if labels.isna().any():
        unknown = labels_df[category_col][labels.isna()].unique()
        raise ValueError(
            f"Unknown categories found: {unknown.tolist()}. "
            f"Valid categories: {list(category_mapping.keys())}"
        )
    
    return labels.astype(int).values


def load_features_and_labels(features_path, target='binary', verbose=True):
    """
    Load features and labels with standardized format.
    
    This function handles multiple file locations and naming conventions:
    1. features_path/target/features.parquet + labels.parquet
    2. features_path/features.parquet + labels.parquet
    
    Args:
        features_path: Path to features directory
        target: Target type ('binary', 'primary', 'sub')
        verbose: Print loading information
    
    Returns:
        tuple: (X_df, y, labels_df)
            - X_df: Features DataFrame with series_id as index
            - y: Target labels as numpy array
            - labels_df: Full labels DataFrame with series_id as index
    
    Raises:
        FileNotFoundError: If features/labels not found
        ValueError: If required columns missing
    """
    features_path = Path(features_path)
    
    # Try multiple locations
    candidates = [
        (features_path / target, f"target subdirectory '{target}'"),
        (features_path, "root directory"),
    ]
    
    feat_file = None
    lab_file = None
    source_desc = None
    
    for path, desc in candidates:
        f = path / 'features.parquet'
        l = path / 'labels.parquet'
        if f.exists() and l.exists():
            feat_file = f
            lab_file = l
            source_desc = desc
            break
    
    if feat_file is None:
        raise FileNotFoundError(
            f"Features not found in: {features_path}\n"
            f"Tried locations:\n" +
            "\n".join(f"  - {p}" for p, _ in candidates)
        )
    
    if verbose:
        print(f"Loading from {source_desc}")
        print(f"  Features: {feat_file}")
        print(f"  Labels:   {lab_file}")
    
    # Load files
    X_df = pd.read_parquet(feat_file)
    labels_df = pd.read_parquet(lab_file)
    
    if verbose:
        print(f"  Features shape: {X_df.shape}")
        print(f"  Labels shape:   {labels_df.shape}")
    
    # Standardize identifier column and set as index
    X_df = standardize_identifier_column(X_df, 'series_id', set_as_index=True)
    labels_df = standardize_identifier_column(labels_df, 'series_id', set_as_index=True)
    
    # Align indices (inner join)
    X_df, labels_df = X_df.align(labels_df, join='inner', axis=0)
    
    if verbose:
        print(f"  Aligned samples: {len(X_df):,}")
    
    # Extract labels based on target type
    if target == 'binary':
        y = extract_binary_labels(labels_df)
    elif target == 'primary':
        y = extract_multiclass_labels(labels_df, PRIMARY_CATEGORY_MAPPING, 'primary_category')
    elif target == 'sub':
        # Sub-category requires custom mapping per primary category
        raise NotImplementedError("Sub-category classification not yet implemented")
    else:
        raise ValueError(f"Unknown target type: {target}. Must be 'binary', 'primary', or 'sub'")
    
    if verbose:
        # Print class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n  Class distribution:")
        for label, count in zip(unique, counts):
            pct = 100 * count / len(y)
            print(f"    {label}: {count:6,} ({pct:5.1f}%)")
    
    return X_df, y, labels_df


def split_train_test(X, y, sample_ids=None, test_size=None, random_state=None, 
                     stratify=True):
    """
    Split data into train and test sets with consistent defaults.
    
    Args:
        X: Features (DataFrame or array)
        y: Labels (array)
        sample_ids: Sample identifiers (optional)
        test_size: Test set size (default: DEFAULT_TEST_SIZE)
        random_state: Random state (default: DEFAULT_RANDOM_STATE)
        stratify: Use stratified split (default: True)
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, train_ids, test_ids)
               If sample_ids is None, returns (X_train, X_test, y_train, y_test)
    """
    if test_size is None:
        test_size = DEFAULT_TEST_SIZE
    
    if random_state is None:
        random_state = DEFAULT_RANDOM_STATE
    
    stratify_arg = y if stratify else None
    
    if sample_ids is not None:
        X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(
            X, y, sample_ids,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arg
        )
        return X_train, X_test, y_train, y_test, train_ids, test_ids
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arg
        )
        return X_train, X_test, y_train, y_test


def remove_series_id_leakage(df):
    """
    Remove series_id from DataFrame to prevent data leakage.
    
    This is critical for training to ensure the model doesn't learn from IDs.
    
    Args:
        df: DataFrame potentially containing series_id
    
    Returns:
        DataFrame without series_id (as column or index)
    """
    df = df.copy()
    
    # If series_id is in index, reset and drop
    if df.index.name == 'series_id' or ('series_id' in getattr(df.index, 'names', [])):
        df = df.reset_index()
    
    # If series_id is a column, drop it
    if 'series_id' in df.columns:
        df = df.drop(columns=['series_id'])
    
    return df
