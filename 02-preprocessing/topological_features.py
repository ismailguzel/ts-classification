"""
Topological Feature Extraction for Time Series Classification.

Fast parallel implementation using Ripser and joblib.

Usage:
    python topological_features.py --input ../data/raw/unified-5k --output ../data/features/unified-5k/topological --n-jobs 8
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
    from ripser import ripser
    from joblib import Parallel, delayed
except ImportError:
    print("Required packages not installed.")
    print("Run: pip install ripser joblib")
    sys.exit(1)


class TopologicalFeatureExtractor:
    """
    Fast topological feature extractor using Ripser with parallel processing.
    
    Parameters:
        embedding_dim (int): Takens embedding dimension (default: 3)
        embedding_delay (int): Time delay for embedding (default: 1)
        homology_dims (tuple): Homology dimensions, e.g. (0, 1) (default: (0, 1))
        n_landscapes (int): Number of persistence landscapes (default: 5)
        landscape_resolution (int): Landscape resolution (default: 100)
        normalize (bool): Normalize features (default: True)
        max_thresh (float): Maximum filtration value for Ripser (default: inf)
        n_perm (int): Number of permutations for Ripser (default: None, uses all points)
        use_mean_landscape (bool): Use mean of all landscapes instead of concatenating (default: False)
    """
    
    def __init__(
        self,
        embedding_dim=3,
        embedding_delay=1,
        homology_dims=(0, 1),
        n_landscapes=5,
        landscape_resolution=100,
        normalize=True,
        max_thresh=np.inf,
        n_perm=None,
        use_mean_landscape=False
    ):
        self.embedding_dim = embedding_dim
        self.embedding_delay = embedding_delay
        self.homology_dims = homology_dims
        self.n_landscapes = n_landscapes
        self.landscape_resolution = landscape_resolution
        self.normalize = normalize
        self.max_thresh = max_thresh
        self.n_perm = n_perm
        self.use_mean_landscape = use_mean_landscape
        
    def takens_embedding(self, ts):
        """Time delay embedding."""
        if isinstance(ts, pd.Series):
            ts = ts.values
        ts = np.asarray(ts).flatten()
        
        min_len = self.embedding_dim * self.embedding_delay
        if len(ts) < min_len:
            raise ValueError(f"Time series too short: {len(ts)} < {min_len}")
        
        n = len(ts) - (self.embedding_dim - 1) * self.embedding_delay
        embedded = np.zeros((n, self.embedding_dim))
        
        for i in range(self.embedding_dim):
            start = i * self.embedding_delay
            embedded[:, i] = ts[start:start + n]
        
        return embedded
    
    def compute_persistence_diagrams(self, ts):
        """Compute persistence diagrams using Ripser."""
        point_cloud = self.takens_embedding(ts)
        
        # Run Ripser with threshold and optional n_perm to speed up
        ripser_kwargs = {
            'maxdim': max(self.homology_dims),
            'thresh': self.max_thresh
        }
        
        # Only add n_perm if specified (not None)
        if self.n_perm is not None:
            ripser_kwargs['n_perm'] = self.n_perm
        
        result = ripser(point_cloud, **ripser_kwargs)
        
        diagrams = {}
        for dim in self.homology_dims:
            if dim < len(result['dgms']):
                diagrams[dim] = result['dgms'][dim]
            else:
                diagrams[dim] = np.array([[0, 0]])
        
        return diagrams
    
    def compute_persistence_landscape(self, diagram):
        """
        Fast persistence landscape computation.
        
        Returns:
            If use_mean_landscape=False: Flattened array (n_landscapes * resolution,)
            If use_mean_landscape=True: Mean landscape array (resolution,)
        """
        # Clean diagram
        diagram = diagram[np.isfinite(diagram).all(axis=1)]
        diagram = diagram[diagram[:, 1] > diagram[:, 0]]
        
        # Determine output size
        if self.use_mean_landscape:
            output_size = self.landscape_resolution
        else:
            output_size = self.n_landscapes * self.landscape_resolution
        
        if len(diagram) == 0:
            return np.zeros(output_size)
        
        min_birth = diagram[:, 0].min()
        max_death = diagram[:, 1].max()
        
        if max_death <= min_birth:
            return np.zeros(output_size)
        
        # Grid
        grid = np.linspace(min_birth, max_death, self.landscape_resolution)
        landscapes = np.zeros((self.n_landscapes, self.landscape_resolution))
        
        # Vectorized computation
        births = diagram[:, 0]
        deaths = diagram[:, 1]
        
        for i, x in enumerate(grid):
            # Tent function values for all bars
            mask = (births <= x) & (x <= deaths)
            if mask.any():
                lambda_vals = np.minimum(x - births[mask], deaths[mask] - x)
                lambda_vals = np.sort(lambda_vals)[::-1]  # Descending
                
                n_vals = min(self.n_landscapes, len(lambda_vals))
                landscapes[:n_vals, i] = lambda_vals[:n_vals]
        
        # Return mean or flattened
        if self.use_mean_landscape:
            return landscapes.mean(axis=0)  # Shape: (resolution,)
        else:
            return landscapes.flatten()  # Shape: (n_landscapes * resolution,)
    
    def diagrams_to_features(self, diagrams):
        """Convert diagrams to feature vector."""
        features = []
        for dim in sorted(diagrams.keys()):
            landscape_feats = self.compute_persistence_landscape(diagrams[dim])
            features.append(landscape_feats)
        
        features = np.concatenate(features)
        
        if self.normalize and features.max() > 0:
            features = features / features.max()
        
        return features
    
    def extract_features(self, ts):
        """Extract topological features from a single time series."""
        diagrams = self.compute_persistence_diagrams(ts)
        return self.diagrams_to_features(diagrams)
    
    def _process_single_series(self, series_id, ts_data):
        """Process a single series (for parallel execution)."""
        try:
            features = self.extract_features(ts_data)
            return series_id, features, None
        except Exception as e:
            return series_id, None, str(e)
    
    def extract_batch(self, series_dict, n_jobs=1, verbose=True):
        """
        Extract features from multiple series in parallel.
        
        Args:
            series_dict: Dictionary {series_id: time_series_array}
            n_jobs: Number of parallel jobs
            verbose: Show progress bar
            
        Returns:
            features_df: DataFrame with features
            failed_ids: List of failed series IDs
        """
        series_ids = list(series_dict.keys())
        series_data = [series_dict[sid] for sid in series_ids]
        
        if verbose:
            print(f"  Processing {len(series_ids)} series with {n_jobs} jobs...")
        
        # Parallel processing
        if n_jobs == 1:
            # Sequential
            results = []
            for sid, ts in tqdm(zip(series_ids, series_data), 
                               total=len(series_ids),
                               desc="  Extracting features",
                               disable=not verbose):
                results.append(self._process_single_series(sid, ts))
        else:
            # Parallel
            results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(self._process_single_series)(sid, ts)
                for sid, ts in tqdm(zip(series_ids, series_data),
                                   total=len(series_ids),
                                   desc="  Extracting features",
                                   disable=not verbose)
            )
        
        # Collect results
        feature_list = []
        id_list = []
        failed_ids = []
        
        for sid, feats, error in results:
            if error is None:
                feature_list.append(feats)
                id_list.append(sid)
            else:
                failed_ids.append((sid, error))
        
        if failed_ids and verbose:
            print(f"  ⚠ {len(failed_ids)} series failed")
        
        # Create DataFrame
        if not feature_list:
            raise ValueError("No features extracted successfully")
        
        feature_names = self.get_feature_names()
        features_df = pd.DataFrame(feature_list, columns=feature_names)
        features_df['series_id'] = id_list
        
        return features_df, failed_ids
    
    def get_feature_names(self):
        """Generate feature names."""
        names = []
        if self.use_mean_landscape:
            # Mean landscape: one feature per bin per homology dimension
            for dim in self.homology_dims:
                for i in range(self.landscape_resolution):
                    names.append(f"topo_H{dim}_mean_bin{i}")
        else:
            # Regular: n_landscapes features per bin per homology dimension
            for dim in self.homology_dims:
                for k in range(self.n_landscapes):
                    for i in range(self.landscape_resolution):
                        names.append(f"topo_H{dim}_L{k}_bin{i}")
        return names
    
    def get_parquet_files(self, input_path):
        """Find all parquet files."""
        input_path = Path(input_path)
        if not input_path.exists():
            raise ValueError(f"Path not found: {input_path}")
        
        files = sorted(input_path.glob("**/*.parquet"))
        if not files:
            raise ValueError(f"No parquet files in: {input_path}")
        
        print(f"Found {len(files)} parquet files")
        return files
    
    def load_chunk(self, chunk_files):
        """Load and clean a chunk of parquet files."""
        dfs = [pd.read_parquet(f) for f in chunk_files]
        df = pd.concat(dfs, ignore_index=True)
        
        if 'series_id' not in df.columns or 'data' not in df.columns:
            raise ValueError("Missing required columns: series_id, data")
        
        # Clean NaN
        if df['data'].isna().sum() > 0:
            df['data'] = df.groupby('series_id')['data'].ffill().bfill()
            df = df.dropna(subset=['data'])
        
        # Clean inf
        if np.isinf(df['data']).sum() > 0:
            finite = df.loc[np.isfinite(df['data']), 'data']
            if len(finite) > 0:
                df.loc[df['data'] == np.inf, 'data'] = finite.max()
                df.loc[df['data'] == -np.inf, 'data'] = finite.min()
            else:
                df = df[np.isfinite(df['data'])]
        
        return df
    
    def process_chunk_by_chunk(self, input_path, output_path, chunk_size=10, 
                               max_files=None, n_jobs=1):
        """
        Process data in chunks with parallel feature extraction.
        
        Args:
            input_path: Input directory
            output_path: Output directory
            chunk_size: Files per chunk
            max_files: Max files to process (None = all)
            n_jobs: Parallel jobs for feature extraction
        """
        import gc
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        temp_dir = output_path / 'temp_chunks'
        temp_dir.mkdir(exist_ok=True)
        
        files = self.get_parquet_files(input_path)
        if max_files:
            files = files[:max_files]
            print(f"Limiting to {max_files} files")
        
        num_chunks = (len(files) + chunk_size - 1) // chunk_size
        print(f"\n{'='*80}")
        print(f"Processing {len(files)} files in {num_chunks} chunks")
        print(f"Chunk size: {chunk_size} files, Parallel jobs: {n_jobs}")
        print(f"{'='*80}\n")
        
        all_labels = []
        
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, len(files))
            chunk_files = files[start:end]
            
            print(f"\n{'─'*80}")
            print(f"Chunk {chunk_idx + 1}/{num_chunks}: Files {start+1}-{end}")
            print(f"{'─'*80}")
            
            # Load
            print(f"[1/3] Loading {len(chunk_files)} files...")
            df = self.load_chunk(chunk_files)
            n_series = df['series_id'].nunique()
            print(f"      ✓ {len(df)} rows, {n_series} series")
            
            # Convert to dict for parallel processing
            print(f"[2/3] Extracting topological features (Ripser + parallel)...")
            series_dict = {sid: group['data'].values 
                          for sid, group in df.groupby('series_id')}
            
            features_df, failed = self.extract_batch(series_dict, n_jobs=n_jobs)
            print(f"      ✓ {features_df.shape[1]-1} features for {features_df.shape[0]} series")
            
            # Labels
            labels_df = df.groupby('series_id').agg({
                'is_stationary': 'first',
                'primary_category': 'first',
                'sub_category': 'first'
            }).reset_index()
            all_labels.append(labels_df)
            
            # Save
            chunk_file = temp_dir / f'chunk_{chunk_idx:04d}.parquet'
            print(f"[3/3] Saving {chunk_file.name}")
            features_df.to_parquet(chunk_file, index=False)
            size_mb = chunk_file.stat().st_size / 1024**2
            print(f"      ✓ Saved ({size_mb:.1f} MB)")
            
            # Cleanup
            del df, series_dict, features_df
            gc.collect()
        
        # Combine
        print(f"\n{'='*80}")
        print(f"Combining {num_chunks} chunks...")
        print(f"{'='*80}\n")
        
        chunk_files = sorted(temp_dir.glob('chunk_*.parquet'))
        print(f"Loading {len(chunk_files)} chunk files...")
        
        chunks = [pd.read_parquet(cf) for cf in tqdm(chunk_files, desc="Loading")]
        features = pd.concat(chunks, ignore_index=True)
        features = features.set_index('series_id')
        print(f"✓ Combined features: {features.shape}")
        
        labels = pd.concat(all_labels, ignore_index=True)
        print(f"✓ Combined labels: {labels.shape}")
        
        # Save final
        self.save_features(features, labels, output_path)
        
        # Cleanup temp
        print(f"\nCleaning up...")
        for cf in chunk_files:
            cf.unlink()
        temp_dir.rmdir()
        print(f"✓ Temp files removed")
        
        return features, labels
    
    def save_features(self, features, labels, output_path):
        """Save features and labels."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Features
        features_file = output_path / 'features.parquet'
        print(f"\nSaving: {features_file}")
        features_copy = features.copy()
        if features_copy.index.name != 'series_id':
            features_copy.index.name = 'series_id'
        features_copy.reset_index().to_parquet(features_file, index=False)
        
        # Labels
        labels_file = output_path / 'labels.parquet'
        print(f"Saving: {labels_file}")
        labels.to_parquet(labels_file, index=False)
        
        # Feature names
        names_file = output_path / 'feature_names.txt'
        print(f"Saving: {names_file}")
        with open(names_file, 'w') as f:
            f.write('\n'.join(features.columns))
        
        print(f"\n✅ Saved successfully!")
        print(f"   Features: {features.shape}")
        print(f"   Labels: {labels.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Fast topological feature extraction with Ripser"
    )
    parser.add_argument("--input", type=str, default="../data/raw/unified-5k",
                       help="Input directory")
    parser.add_argument("--output", type=str, default="../data/features/unified-5k/topological",
                       help="Output directory")
    parser.add_argument("--embedding-dim", type=int, default=3,
                       help="Takens embedding dimension")
    parser.add_argument("--embedding-delay", type=int, default=1,
                       help="Takens embedding delay")
    parser.add_argument("--n-landscapes", type=int, default=5,
                       help="Number of persistence landscapes")
    parser.add_argument("--landscape-resolution", type=int, default=100,
                       help="Landscape resolution")
    parser.add_argument("--homology-dims", type=str, default="0,1",
                       help="Homology dimensions (comma-separated)")
    parser.add_argument("--max-thresh", type=float, default=np.inf,
                       help="Max filtration threshold (lower = faster)")
    parser.add_argument("--chunk-size", type=int, default=10,
                       help="Files per chunk")
    parser.add_argument("--max-files", type=int, default=None,
                       help="Max files to process")
    parser.add_argument("--n-jobs", type=int, default=1,
                       help="Number of parallel jobs")
    parser.add_argument("--n-perm", type=int, default=None,
                       help="Number of points for Ripser (subsampling for speed)")
    parser.add_argument("--use-mean-landscape", action="store_true",
                       help="Use mean of landscapes (reduces features by n_landscapes factor)")
    parser.add_argument("--no-normalize", action="store_true",
                       help="Disable normalization")
    
    args = parser.parse_args()
    
    homology_dims = tuple(int(d) for d in args.homology_dims.split(','))
    
    print("="*80)
    print("Topological Feature Extraction (Fast Parallel)")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Embedding: dim={args.embedding_dim}, delay={args.embedding_delay}")
    print(f"Homology: {homology_dims}")
    print(f"Landscapes: {args.n_landscapes} x {args.landscape_resolution}")
    if args.use_mean_landscape:
        total_features = len(homology_dims) * args.landscape_resolution
        print(f"Use mean landscape: True (→ {total_features} features)")
    else:
        total_features = len(homology_dims) * args.n_landscapes * args.landscape_resolution
        print(f"Use mean landscape: False (→ {total_features} features)")
    print(f"Max threshold: {args.max_thresh}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Parallel jobs: {args.n_jobs}")
    if args.n_perm:
        print(f"Ripser n_perm: {args.n_perm}")
    if args.max_files:
        print(f"Max files: {args.max_files}")
    print("="*80)
    
    extractor = TopologicalFeatureExtractor(
        embedding_dim=args.embedding_dim,
        embedding_delay=args.embedding_delay,
        homology_dims=homology_dims,
        n_landscapes=args.n_landscapes,
        landscape_resolution=args.landscape_resolution,
        normalize=not args.no_normalize,
        max_thresh=args.max_thresh,
        n_perm=args.n_perm,
        use_mean_landscape=args.use_mean_landscape
    )
    
    features, labels = extractor.process_chunk_by_chunk(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
        max_files=args.max_files,
        n_jobs=args.n_jobs
    )
    
    print("\n🎉 Completed!")
    print(f"   Features: {features.shape}")
    print(f"   Labels: {labels.shape}")


if __name__ == "__main__":
    main()
