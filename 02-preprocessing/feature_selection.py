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
    
    def remove_low_variance(self, X, threshold=0.01):
        """Remove features with low variance."""
        print(f"\nRemoving low variance features (threshold={threshold})...")
        print(f"Original features: {X.shape[1]}")
        
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)
        
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
    
    def select_statistical(self, X, y, k=100):
        """Select features using statistical tests (ANOVA F-test)."""
        print(f"\nSelecting top {k} features using ANOVA F-test...")
        print(f"Original features: {X.shape[1]}")
        
        selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()]
        
        print(f"Selected features: {len(selected_features)}")
        
        return X[selected_features], selected_features.tolist()
    
    def select_mutual_info(self, X, y, k=100):
        """Select features using mutual information."""
        print(f"\nSelecting top {k} features using mutual information...")
        print(f"Original features: {X.shape[1]}")
        
        selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()]
        
        print(f"Selected features: {len(selected_features)}")
        
        return X[selected_features], selected_features.tolist()
    
    def select_importance(self, X, y, k=100):
        """Select features using Random Forest feature importance."""
        print(f"\nSelecting top {k} features using Random Forest importance...")
        print(f"Original features: {X.shape[1]}")
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importance
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:min(k, X.shape[1])]
        
        selected_features = X.columns[indices].tolist()
        
        print(f"Selected features: {len(selected_features)}")
        
        return X[selected_features], selected_features
    
    def select_features(self, X, y, target='binary'):
        """
        Select features using specified method.
        
        Args:
            X: Feature matrix
            y: Target labels
            target: Target type ('binary', 'primary', 'sub')
        """
        print("="*80)
        print(f"Feature Selection - Method: {self.method}")
        print(f"Target: {target}")
        print("="*80)
        
        if self.method == 'variance':
            X_selected, features = self.remove_low_variance(X, threshold=0.01)
            X_selected, features = self.remove_correlated(X_selected, threshold=0.95)
        
        elif self.method == 'correlation':
            X_selected, features = self.remove_correlated(X, threshold=0.90)
        
        elif self.method == 'statistical':
            # First remove low variance
            X_filtered, _ = self.remove_low_variance(X, threshold=0.001)
            X_selected, features = self.select_statistical(X_filtered, y, k=self.n_features)
        
        elif self.method == 'mutual_info':
            # First remove low variance
            X_filtered, _ = self.remove_low_variance(X, threshold=0.001)
            X_selected, features = self.select_mutual_info(X_filtered, y, k=self.n_features)
        
        elif self.method == 'importance':
            # First remove low variance
            X_filtered, _ = self.remove_low_variance(X, threshold=0.001)
            X_selected, features = self.select_importance(X_filtered, y, k=self.n_features)
        
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
    parser.add_argument('--input', type=str, default='../data/features/unified-5k/allfeatures',
                        help='Input directory with extracted features')
    parser.add_argument('--output', type=str, default='../data/features/unified-5k/selected',
                        help='Output directory for selected features')
    parser.add_argument('--method', type=str, default='mutual_info',
                        choices=['variance', 'correlation', 'statistical', 'mutual_info', 'importance'],
                        help='Feature selection method')
    parser.add_argument('--n-features', type=int, default=100,
                        help='Number of features to select')
    parser.add_argument('--target', type=str, default='binary',
                        choices=['binary', 'primary', 'sub', 'all'],
                        help='Target variable for selection')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Feature Selection")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Method: {args.method}")
    print(f"N features: {args.n_features}")
    print(f"Target: {args.target}")
    print("="*80)
    
    # Initialize selector
    selector = FeatureSelector(method=args.method, n_features=args.n_features)
    
    # Load data
    features_df, labels_df = selector.load_data(args.input)
    
    # Encode labels
    le = LabelEncoder()
    
    # Select features for each target
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
        
        # Select features
        X_selected, selected_features = selector.select_features(features_df, y, target)
        
        # Save results
        selector.save_selected_features(X_selected, labels_df, selected_features, args.output, target)
    
    print("\nFeature selection completed successfully!")


if __name__ == "__main__":
    main()
