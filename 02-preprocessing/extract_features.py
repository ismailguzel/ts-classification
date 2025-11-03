"""
Extract TSFresh features for hierarchical classification.

Run from the `02-preprocessing` directory.

Usage:
    python extract_features.py --input ../data/raw/unified-90k --output ../data/features

Outputs under the specified `--output` directory:
    - features.parquet
    - labels.parquet
    - feature_names.txt
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
    
    def __init__(self, feature_set='minimal', n_jobs=110):
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
    
    def get_parquet_files(self, input_path):
        """Find all parquet files in the input directory."""
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")
        
        print(f"Scanning for parquet files in: {input_path}")
        
        # Find all parquet files (including subdirectories) and sort for deterministic ordering
        parquet_files = sorted(input_path.glob("**/*.parquet"))
        
        if not parquet_files:
            raise ValueError(f"No parquet files found in: {input_path}")
        
        print(f"Found {len(parquet_files)} parquet files")
        return parquet_files
    
    def load_and_clean_chunk(self, chunk_files):
        """
        Load and clean a chunk of parquet files.
        
        Args:
            chunk_files: List of parquet file paths
            
        Returns:
            DataFrame with columns: series_id, time, data, is_stationary, primary_category, sub_category
        """
        # Load chunk files
        dfs = []
        for file_idx, file in enumerate(chunk_files):
            df = pd.read_parquet(file)
            # IDs are guaranteed unique at source via start_id; keep as-is
            dfs.append(df)
        
        # Combine chunk dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Check for required columns
        if 'series_id' not in combined_df.columns:
            raise ValueError("Column 'series_id' not found in data")
        if 'data' not in combined_df.columns:
            raise ValueError("Column 'data' not found in data")
        
        # Clean NaN values (TSFresh doesn't accept NaN)
        nan_count_before = combined_df['data'].isna().sum()
        if nan_count_before > 0:
            combined_df['data'] = combined_df.groupby('series_id')['data'].ffill().bfill()
            nan_count_after = combined_df['data'].isna().sum()
            if nan_count_after > 0:
                combined_df = combined_df.dropna(subset=['data'])
        
        # Clean infinite values
        inf_count = np.isinf(combined_df['data']).sum()
        if inf_count > 0:
            finite_values = combined_df.loc[np.isfinite(combined_df['data']), 'data']
            if len(finite_values) > 0:
                max_finite = finite_values.max()
                min_finite = finite_values.min()
                combined_df.loc[combined_df['data'] == np.inf, 'data'] = max_finite
                combined_df.loc[combined_df['data'] == -np.inf, 'data'] = min_finite
            else:
                combined_df = combined_df[np.isfinite(combined_df['data'])]
        
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
        # Extract features
        features = extract_features(
            timeseries_df,
            column_id='id',
            column_sort='sort',
            column_value='value',
            default_fc_parameters=self.extraction_settings,
            n_jobs=self.n_jobs,
            show_warnings=False,
            disable_progressbar=False,
            chunksize=500  # Process 500 series at a time within TSFresh
        )
        
        # Impute missing values
        features = impute(features)
        
        return features
    
    def process_chunk_by_chunk(self, input_path, output_path, chunk_size=10, max_files=None):
        """
        Process data chunk by chunk to avoid RAM overflow.
        
        Args:
            input_path: Path to directory containing parquet files
            output_path: Directory to save intermediate and final features
            chunk_size: Number of parquet files per chunk
            max_files: Maximum number of files to process (None = all)
        """
        import gc
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create temp directory for chunk results
        temp_dir = output_path / 'temp_chunks'
        temp_dir.mkdir(exist_ok=True)
        
        # Get all parquet files
        parquet_files = self.get_parquet_files(input_path)
        
        # Limit files if requested
        if max_files is not None:
            parquet_files = parquet_files[:max_files]
            print(f"Limiting to first {max_files} files")
        
        # Process in chunks
        num_chunks = (len(parquet_files) + chunk_size - 1) // chunk_size
        print(f"\n{'='*80}")
        print(f"Processing {len(parquet_files)} files in {num_chunks} chunks (chunk_size={chunk_size})")
        print(f"{'='*80}\n")
        
        all_labels = []
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(parquet_files))
            chunk_files = parquet_files[start_idx:end_idx]
            
            print(f"\n{'─'*80}")
            print(f"Chunk {chunk_idx + 1}/{num_chunks}: Processing files {start_idx+1}-{end_idx}")
            print(f"{'─'*80}")
            
            # Load and clean chunk
            print(f"  [1/4] Loading {len(chunk_files)} parquet files...")
            chunk_df = self.load_and_clean_chunk(chunk_files)
            print(f"        ✓ Loaded {len(chunk_df)} rows, {chunk_df['series_id'].nunique()} series")
            
            # Prepare for TSFresh
            print(f"  [2/4] Preparing data for TSFresh...")
            timeseries_df, labels_df = self.prepare_for_tsfresh(chunk_df)
            all_labels.append(labels_df)
            print(f"        ✓ Prepared {len(timeseries_df)} rows")
            
            # Extract features
            print(f"  [3/4] Extracting features (TSFresh with {self.feature_set} settings)...")
            chunk_features = self.extract_features(timeseries_df)
            print(f"        ✓ Extracted {chunk_features.shape[1]} features for {chunk_features.shape[0]} series")
            
            # Save chunk to disk immediately
            chunk_file = temp_dir / f'features_chunk_{chunk_idx:04d}.parquet'
            print(f"  [4/4] Saving chunk to disk: {chunk_file.name}")
            chunk_features.reset_index().to_parquet(chunk_file, index=False)
            print(f"        ✓ Chunk saved ({chunk_features.memory_usage(deep=True).sum() / 1024**2:.1f} MB)")
            
            # Clean up RAM immediately
            del chunk_df, timeseries_df, labels_df, chunk_features
            gc.collect()
            print(f"        ✓ Memory cleaned")
        
        # Combine all chunks
        print(f"\n{'='*80}")
        print(f"Combining {num_chunks} chunks...")
        print(f"{'='*80}\n")
        
        chunk_files = sorted(temp_dir.glob('features_chunk_*.parquet'))
        print(f"  Found {len(chunk_files)} chunk files")
        
        # Read and combine features
        print(f"  Combining feature chunks...")
        feature_chunks = []
        for cf in tqdm(chunk_files, desc="Loading chunks"):
            feature_chunks.append(pd.read_parquet(cf))
        
        features = pd.concat(feature_chunks, ignore_index=False)
        features = features.set_index('index') if 'index' in features.columns else features
        print(f"  ✓ Combined features shape: {features.shape}")
        
        # Combine labels
        labels_df = pd.concat(all_labels, ignore_index=True)
        print(f"  ✓ Combined labels shape: {labels_df.shape}")
        
        # Save final results
        self.save_features(features, labels_df, output_path)
        
        # Clean up temp directory
        print(f"\n  Cleaning up temporary files...")
        for cf in chunk_files:
            cf.unlink()
        temp_dir.rmdir()
        print(f"  ✓ Temp directory removed")
        
        return features, labels_df
    
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
    parser.add_argument('--input', type=str, default='../data/raw/unified-90k',
                        help='Input directory with parquet files')
    parser.add_argument('--output', type=str, default='../data/features',
                        help='Output directory for features')
    parser.add_argument('--feature-set', type=str, default='efficient',
                        choices=['minimal', 'efficient', 'comprehensive'],
                        help='Feature set to extract')
    parser.add_argument('--n-jobs', type=int, default=110,
                        help='Number of parallel jobs for TSFresh')
    parser.add_argument('--chunk-size', type=int, default=10,
                        help='Number of parquet files per chunk (lower = less RAM)')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to process (for testing)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("TSFresh Feature Extraction (Chunk-by-Chunk)")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Feature set: {args.feature_set}")
    print(f"Parallel jobs: {args.n_jobs}")
    print(f"Chunk size: {args.chunk_size} files/chunk")
    if args.max_files:
        print(f"Max files: {args.max_files}")
    print("="*80)
    
    # Initialize extractor
    extractor = TSFreshFeatureExtractor(
        feature_set=args.feature_set,
        n_jobs=args.n_jobs
    )
    
    # Process chunk by chunk (never load all data to RAM!)
    features, labels_df = extractor.process_chunk_by_chunk(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
        max_files=args.max_files
    )
    
    print("\n🎉 Feature extraction completed successfully!")
    print(f"   Final features: {features.shape}")
    print(f"   Final labels: {labels_df.shape}")


if __name__ == "__main__":
    main()
