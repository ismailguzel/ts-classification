"""
TSFresh Feature Engineering Script
===================================

Extracts time series features using TSFresh library for hierarchical classification.
Run this script from the 02-preprocessing directory.

Usage:
    python extract_features.py --input ../data/raw/unified-150k --output ../data/features

Features:
    - Comprehensive: ~800 features per time series
    - Efficient: Only extracts relevant features (optional feature selection)
    - Optimized: Minimal computation settings for balanced classification

Time: ~2-4 hours for 150K series
Size: ~5-8 GB feature matrix
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

try:
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute
    from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters
    from tsfresh.feature_selection.relevance import calculate_relevance_table
except ImportError:
    print("TSFresh not installed. Installing...")
    print("Run: pip install tsfresh")
    sys.exit(1)


class TSFreshFeatureExtractor:
    """Extract features from time series using TSFresh."""
    
    def __init__(self, feature_set='minimal', n_jobs=4):
        """
        Initialize feature extractor.
        
        Args:
            feature_set: 'minimal', 'comprehensive', or 'efficient'
            n_jobs: Number of parallel jobs
        """
        self.feature_set = feature_set
        self.n_jobs = n_jobs
        
        # Select feature extraction settings
        if feature_set == 'minimal':
            self.extraction_settings = MinimalFCParameters()
        elif feature_set == 'comprehensive':
            self.extraction_settings = ComprehensiveFCParameters()
        elif feature_set == 'efficient':
            # Custom efficient settings for classification
            self.extraction_settings = self._get_efficient_settings()
        else:
            raise ValueError(f"Unknown feature set: {feature_set}")
    
    def _get_efficient_settings(self):
        """Get efficient feature extraction settings."""
        from tsfresh.feature_extraction import EfficientFCParameters
        return EfficientFCParameters()
    
    def load_data(self, input_path):
        """
        Load time series data from parquet files.
        
        Args:
            input_path: Path to directory containing parquet files
            
        Returns:
            DataFrame with columns: series_id, time, data, is_stationary, primary_category, sub_category
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")
        
        print(f"Loading data from: {input_path}")
        
        # Find all parquet files (including subdirectories)
        parquet_files = list(input_path.glob("**/*.parquet"))
        
        if not parquet_files:
            raise ValueError(f"No parquet files found in: {input_path}")
        
        print(f"Found {len(parquet_files)} parquet files")
        
        # Load all files
        dfs = []
        for file in tqdm(parquet_files, desc="Loading files"):
            df = pd.read_parquet(file)
            dfs.append(df)
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        print(f"Loaded {len(combined_df)} total rows")
        print(f"Unique series: {combined_df['series_id'].nunique()}")
        
        # Check for required columns
        if 'series_id' not in combined_df.columns:
            raise ValueError("Column 'series_id' not found in data")
        if 'data' not in combined_df.columns:
            raise ValueError("Column 'data' not found in data")
        
        # Clean NaN values (TSFresh doesn't accept NaN)
        print(f"Checking for NaN values...")
        nan_count_before = combined_df['data'].isna().sum()
        if nan_count_before > 0:
            print(f"  Found {nan_count_before} NaN values in 'data' column")
            print(f"  Filling NaN with forward fill then backward fill...")
            combined_df['data'] = combined_df.groupby('series_id')['data'].ffill().bfill()
            nan_count_after = combined_df['data'].isna().sum()
            if nan_count_after > 0:
                print(f"  Warning: {nan_count_after} NaN values remain, dropping those rows...")
                combined_df = combined_df.dropna(subset=['data'])
            print(f"  ✓ Cleaned {nan_count_before - nan_count_after} NaN values")
        else:
            print(f"  ✓ No NaN values found")
        
        # Clean infinite values
        print(f"Checking for inf values...")
        inf_count = np.isinf(combined_df['data']).sum()
        if inf_count > 0:
            print(f"  Found {inf_count} inf values in 'data' column")
            print(f"  Replacing inf with finite max/min values...")
            # Replace +inf with max finite value, -inf with min finite value
            finite_values = combined_df.loc[np.isfinite(combined_df['data']), 'data']
            if len(finite_values) > 0:
                max_finite = finite_values.max()
                min_finite = finite_values.min()
                combined_df.loc[combined_df['data'] == np.inf, 'data'] = max_finite
                combined_df.loc[combined_df['data'] == -np.inf, 'data'] = min_finite
                print(f"  ✓ Replaced {inf_count} inf values")
            else:
                print(f"  Warning: No finite values found, dropping inf rows...")
                combined_df = combined_df[np.isfinite(combined_df['data'])]
        else:
            print(f"  ✓ No inf values found")
        
        return combined_df
    
    def prepare_for_tsfresh(self, df):
        """
        Prepare data for TSFresh format.
        
        Args:
            df: DataFrame with columns: series_id, time, data, is_stationary, primary_category, sub_category
            
        Returns:
            timeseries_df: DataFrame for feature extraction (series_id, time, data)
            labels_df: DataFrame with labels (series_id, is_stationary, primary_category, sub_category)
        """
        # Extract labels
        labels_df = df.groupby('series_id').agg({
            'is_stationary': 'first',
            'primary_category': 'first',
            'sub_category': 'first'
        }).reset_index()
        
        # Prepare time series data for TSFresh
        # TSFresh expects columns: id, sort (time), value
        timeseries_df = df[['series_id', 'time', 'data']].copy()
        timeseries_df = timeseries_df.rename(columns={
            'series_id': 'id',
            'time': 'sort',
            'data': 'value'
        })
        
        return timeseries_df, labels_df
    
    def extract_features(self, timeseries_df):
        """
        Extract features using TSFresh.
        
        Args:
            timeseries_df: DataFrame with columns: id, sort, value
            
        Returns:
            DataFrame with extracted features (rows=series, columns=features)
        """
        print(f"\nExtracting features using {self.feature_set} settings...")
        print(f"Number of series: {timeseries_df['id'].nunique()}")
        
        # Extract features
        features = extract_features(
            timeseries_df,
            column_id='id',
            column_sort='sort',
            column_value='value',
            default_fc_parameters=self.extraction_settings,
            n_jobs=self.n_jobs,
            show_warnings=False,
            disable_progressbar=False
        )
        
        print(f"Extracted {features.shape[1]} features")
        
        # Impute missing values
        print("Imputing missing values...")
        features = impute(features)
        
        return features
    
    def save_features(self, features, labels_df, output_path):
        """
        Save extracted features and labels.
        
        Args:
            features: DataFrame with extracted features
            labels_df: DataFrame with labels
            output_path: Directory to save features
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save features
        features_file = output_path / 'features.parquet'
        print(f"\nSaving features to: {features_file}")
        features.reset_index().to_parquet(features_file, index=False)
        
        # Save labels
        labels_file = output_path / 'labels.parquet'
        print(f"Saving labels to: {labels_file}")
        labels_df.to_parquet(labels_file, index=False)
        
        # Save feature names
        feature_names_file = output_path / 'feature_names.txt'
        print(f"Saving feature names to: {feature_names_file}")
        with open(feature_names_file, 'w') as f:
            for col in features.columns:
                f.write(f"{col}\n")
        
        print(f"\n✅ Features saved successfully!")
        print(f"   Features shape: {features.shape}")
        print(f"   Labels shape: {labels_df.shape}")


def main():
    parser = argparse.ArgumentParser(description='Extract time series features using TSFresh')
    parser.add_argument('--input', type=str, default='../data/raw/unified-150k',
                        help='Input directory with parquet files')
    parser.add_argument('--output', type=str, default='../data/features',
                        help='Output directory for features')
    parser.add_argument('--feature-set', type=str, default='efficient',
                        choices=['minimal', 'efficient', 'comprehensive'],
                        help='Feature set to extract')
    parser.add_argument('--n-jobs', type=int, default=4,
                        help='Number of parallel jobs')
    
    args = parser.parse_args()
    
    print("="*80)
    print("TSFresh Feature Extraction")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Feature set: {args.feature_set}")
    print(f"Parallel jobs: {args.n_jobs}")
    print("="*80)
    
    # Initialize extractor
    extractor = TSFreshFeatureExtractor(
        feature_set=args.feature_set,
        n_jobs=args.n_jobs
    )
    
    # Load data
    df = extractor.load_data(args.input)
    
    # Prepare for TSFresh
    timeseries_df, labels_df = extractor.prepare_for_tsfresh(df)
    
    # Extract features
    features = extractor.extract_features(timeseries_df)
    
    # Save results
    extractor.save_features(features, labels_df, args.output)
    
    print("\n🎉 Feature extraction completed successfully!")


if __name__ == "__main__":
    main()
