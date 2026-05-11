"""
Select relevant features from TSFresh outputs using multiple methods.

Usage:
    python feature_selection.py --input ../data/features --output ../data/features/selected

Methods:
    - Variance threshold: remove low-variance features
    - Correlation: remove highly correlated features
    - Statistical tests: Chi-square, ANOVA F-test
    - Mutual information: MI-based selection
    - Feature importance: Random Forest importance
"""

import sys
import os
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    mutual_info_classif,
    chi2
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class FeatureSelector:
    """Select relevant features using multiple methods."""
    
    def __init__(self, method='mutual_info', n_features=100):
        """
        Initialize feature selector.
        
        Args:
            method: Selection method ('variance', 'correlation', 'statistical', 'mutual_info', 'importance')
            n_features: Number of features to select
        """
        self.method = method
        self.n_features = n_features
        self.selected_features_ = None
    
    def load_data(self, input_path):
        """
        Load features and labels.
        """
        input_path = Path(input_path)
        
        features_file = input_path / 'features.parquet'
        labels_file = input_path / 'labels.parquet'
        
        print(f"Loading features from: {features_file}")
        features_df = pd.read_parquet(features_file)
        
        print(f"Loading labels from: {labels_file}")
        labels_df = pd.read_parquet(labels_file)
        
        # Standardize identifier column name
        if 'series_id' in features_df.columns:
            features_df = features_df.set_index('series_id')
        elif 'id' in features_df.columns:
            features_df = features_df.rename(columns={'id': 'series_id'}).set_index('series_id')
        elif 'index' in features_df.columns:
            features_df = features_df.rename(columns={'index': 'series_id'}).set_index('series_id')

        if 'series_id' in labels_df.columns:
            labels_df = labels_df.set_index('series_id')
        elif 'id' in labels_df.columns:
            labels_df = labels_df.rename(columns={'id': 'series_id'}).set_index('series_id')
        elif 'index' in labels_df.columns:
            labels_df = labels_df.rename(columns={'index': 'series_id'}).set_index('series_id')

        return features_df, labels_df
    
    def remove_low_variance(self, X, threshold=0.01, X_fit=None):
        """Remove features with low variance.

        X_fit: subset used to fit the threshold (train rows). If None, fits on X.
        """
        print(f"\nRemoving low variance features (threshold={threshold})...")
        print(f"Original features: {X.shape[1]}")

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X_fit if X_fit is not None else X)

        selected_features = X.columns[selector.get_support()]

        print(f"Selected features: {len(selected_features)}")
        print(f"Removed: {X.shape[1] - len(selected_features)}")

        return X[selected_features], selected_features.tolist()
    
    def remove_correlated(self, X, threshold=0.95):
        """Remove highly correlated features."""
        print(f"\nRemoving correlated features (threshold={threshold})...")
        print(f"Original features: {X.shape[1]}")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Select upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        selected_features = [col for col in X.columns if col not in to_drop]
        
        print(f"Selected features: {len(selected_features)}")
        print(f"Removed: {len(to_drop)}")
        
        return X[selected_features], selected_features
    
    def select_statistical(self, X, y, k=100, X_fit=None, y_fit=None):
        """Select features using statistical tests (ANOVA F-test).

        X_fit / y_fit: train subset used to fit the F-test. If None, fits on X/y.
        """
        print(f"\nSelecting top {k} features using ANOVA F-test...")
        print(f"Original features: {X.shape[1]}")

        selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
        selector.fit(X_fit if X_fit is not None else X,
                     y_fit if y_fit is not None else y)

        selected_features = X.columns[selector.get_support()]

        print(f"Selected features: {len(selected_features)}")

        return X[selected_features], selected_features.tolist()
    
    def select_mutual_info(self, X, y, k=100, X_fit=None, y_fit=None):
        """Select features using mutual information.

        X_fit / y_fit: train subset used to fit MI scores. If None, fits on X/y.
        The returned X still contains all rows of X (train + test), filtered to
        the selected feature columns only.
        """
        print(f"\nSelecting top {k} features using mutual information...")
        print(f"Original features: {X.shape[1]}")

        selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
        selector.fit(X_fit if X_fit is not None else X,
                     y_fit if y_fit is not None else y)

        selected_features = X.columns[selector.get_support()]

        print(f"Selected features: {len(selected_features)}")

        return X[selected_features], selected_features.tolist()
    
    def select_importance(self, X, y, k=100, X_fit=None, y_fit=None):
        """Select features using Random Forest feature importance.

        X_fit / y_fit: train subset used to fit the RF. If None, fits on X/y.
        """
        print(f"\nSelecting top {k} features using Random Forest importance...")
        print(f"Original features: {X.shape[1]}")

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_fit if X_fit is not None else X,
               y_fit if y_fit is not None else y)

        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:min(k, X.shape[1])]

        selected_features = X.columns[indices].tolist()

        print(f"Selected features: {len(selected_features)}")

        return X[selected_features], selected_features
    
    def select_features(self, X, y, target='binary', X_train=None, y_train=None):
        """Select features using specified method.

        X_train / y_train: training subset used to fit the selector.
        The returned X contains all rows of X filtered to selected columns —
        train.py will perform the actual train/test split downstream.
        If X_train is None the selector is fit on the full X (legacy behaviour).
        """
        print("="*80)
        print(f"Feature Selection - Method: {self.method}")
        print(f"Target: {target}")
        if X_train is not None:
            print(f"Fitting selector on train subset ({len(X_train)} rows), "
                  f"applying to full dataset ({len(X)} rows).")
        print("="*80)

        if self.method == 'variance':
            X_selected, features = self.remove_low_variance(
                X, threshold=0.01, X_fit=X_train)
            X_selected, features = self.remove_correlated(X_selected, threshold=0.95)

        elif self.method == 'correlation':
            X_selected, features = self.remove_correlated(X, threshold=0.90)

        elif self.method == 'statistical':
            X_filtered, _ = self.remove_low_variance(
                X, threshold=0.001,
                X_fit=X_train)
            X_train_f = X_train[X_filtered.columns] if X_train is not None else None
            X_selected, features = self.select_statistical(
                X_filtered, y, k=self.n_features,
                X_fit=X_train_f, y_fit=y_train)

        elif self.method == 'mutual_info':
            X_filtered, _ = self.remove_low_variance(
                X, threshold=0.001,
                X_fit=X_train)
            X_train_f = X_train[X_filtered.columns] if X_train is not None else None
            X_selected, features = self.select_mutual_info(
                X_filtered, y, k=self.n_features,
                X_fit=X_train_f, y_fit=y_train)

        elif self.method == 'importance':
            X_filtered, _ = self.remove_low_variance(
                X, threshold=0.001,
                X_fit=X_train)
            X_train_f = X_train[X_filtered.columns] if X_train is not None else None
            X_selected, features = self.select_importance(
                X_filtered, y, k=self.n_features,
                X_fit=X_train_f, y_fit=y_train)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.selected_features_ = features

        return X_selected, features
    
    def save_selected_features(self, X, labels_df, features, output_path, target):
        """Save selected features and standardized files."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save legacy selected features file
        features_file = output_path / f'features_{target}_{self.method}.parquet'
        print(f"\nSaving selected features to: {features_file}")
        X.reset_index().to_parquet(features_file, index=False)

        # Save legacy labels file
        labels_file = output_path / f'labels_{target}.parquet'
        if not labels_file.exists():
            print(f"Saving labels to: {labels_file}")
            labels_df.to_parquet(labels_file, index=False)

        # Save feature names
        feature_names_file = output_path / f'feature_names_{target}_{self.method}.txt'
        print(f"Saving feature names to: {feature_names_file}")
        with open(feature_names_file, 'w') as f:
            for feature in features:
                f.write(f"{feature}\n")

        # --- Standardized output for model training ---
        # Save into per-target subdirectory to avoid overwriting when target=all
        std_dir = output_path / target
        std_dir.mkdir(parents=True, exist_ok=True)

        # Save features.parquet (only selected features, with id)
        std_features_file = std_dir / 'features.parquet'
        print(f"Saving standardized features to: {std_features_file}")
        X_std = X.reset_index()
        X_std.to_parquet(std_features_file, index=False)

        # Save labels.parquet (only id and target column)
        std_labels_file = std_dir / 'labels.parquet'
        print(f"Saving standardized labels to: {std_labels_file}")
        
        # Ensure we have the identifier column (either 'id' or 'series_id')
        labels_df_copy = labels_df.copy()
        if 'id' not in labels_df_copy.columns and 'series_id' not in labels_df_copy.columns:
            # Reset index to get the identifier back as a column
            labels_df_copy = labels_df_copy.reset_index()
        
        # Standardize identifier column name to 'series_id'
        if 'id' in labels_df_copy.columns and 'series_id' not in labels_df_copy.columns:
            labels_df_copy = labels_df_copy.rename(columns={'id': 'series_id'})
        elif labels_df_copy.index.name == 'series_id':
            labels_df_copy = labels_df_copy.reset_index()
        
        # Select columns based on target
        if target == 'binary':
            label_cols = ['series_id', 'is_stationary']
        elif target == 'primary':
            label_cols = ['series_id', 'primary_category']
        elif target == 'sub':
            label_cols = ['series_id', 'sub_category']
        else:
            # For 'all', keep all columns with series_id
            label_cols = ['series_id'] + [col for col in labels_df_copy.columns if col != 'series_id']
        
        # Filter only existing columns
        label_cols = [col for col in label_cols if col in labels_df_copy.columns]
        labels_df_std = labels_df_copy[label_cols]
        labels_df_std.to_parquet(std_labels_file, index=False)

        print(f"\nSelected features and standardized files saved successfully!")


def main():
    parser = argparse.ArgumentParser(description='Select relevant features')
    parser.add_argument('--input', type=str, default='../data/features/dataset/allfeatures',
                        help='Input directory with extracted features')
    parser.add_argument('--output', type=str, default='../data/features/dataset/selected',
                        help='Output directory for selected features')
    parser.add_argument('--method', type=str, default='mutual_info',
                        choices=['variance', 'correlation', 'statistical', 'mutual_info', 'importance'],
                        help='Feature selection method')
    parser.add_argument('--n-features', type=int, default=100,
                        help='Number of features to select')
    parser.add_argument('--target', type=str, default='binary',
                        choices=['binary', 'primary', 'sub', 'all'],
                        help='Target variable for selection')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction held out for test (must match train.py, default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for train/test split (must match train.py, default: 42)')

    args = parser.parse_args()

    print("="*80)
    print("Feature Selection")
    print("="*80)
    print(f"Input:        {args.input}")
    print(f"Output:       {args.output}")
    print(f"Method:       {args.method}")
    print(f"N features:   {args.n_features}")
    print(f"Target:       {args.target}")
    print(f"Test size:    {args.test_size}  (selector fit on train split only)")
    print(f"Random state: {args.random_state}")
    print("="*80)

    # Initialize selector
    selector = FeatureSelector(method=args.method, n_features=args.n_features)

    # Load data
    features_df, labels_df = selector.load_data(args.input)

    le = LabelEncoder()

    targets = ['binary', 'primary', 'sub'] if args.target == 'all' else [args.target]

    for target in targets:
        print(f"\n{'='*80}")
        print(f"Processing target: {target}")
        print(f"{'='*80}")

        if target == 'binary':
            y = le.fit_transform(labels_df['is_stationary'])
        elif target == 'primary':
            y = le.fit_transform(labels_df['primary_category'])
        elif target == 'sub':
            y = le.fit_transform(labels_df['sub_category'])

        # Split indices to get training subset for fitting the selector.
        # The full dataset is still saved — train.py performs the real split downstream.
        train_idx, _ = train_test_split(
            np.arange(len(features_df)),
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )
        X_train_fit = features_df.iloc[train_idx]
        y_train_fit = y[train_idx]

        # Select features (selector fit on train, applied to full dataset)
        X_selected, selected_features = selector.select_features(
            features_df, y, target,
            X_train=X_train_fit, y_train=y_train_fit,
        )

        # Save results
        selector.save_selected_features(X_selected, labels_df, selected_features, args.output, target)

    print("\nFeature selection completed successfully!")


if __name__ == "__main__":
    main()
