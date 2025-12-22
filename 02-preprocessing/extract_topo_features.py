"""
Topological Feature Extraction for Time Series

Extracts topological features using persistent homology and saves them
in the features directory structure.

Usage:
    python extract_topo_features.py --data-dir ../data/raw/unified-20k --output-dir ../data/features/unified-20k/topological

Requirements:
    - ripser
    - joblib
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import topological feature extractor
sys.path.append(str(Path(__file__).parent / 'topological-features'))
from topological_features import TopologicalFeatureExtractor


def main():
    parser = argparse.ArgumentParser(description='Extract topological features from time series')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Input directory with raw parquet files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for topological features')
    parser.add_argument('--embedding-dim', type=int, default=3,
                       help='Takens embedding dimension (default: 3)')
    parser.add_argument('--embedding-delay', type=int, default=1,
                       help='Time delay for embedding (default: 1)')
    parser.add_argument('--n-landscapes', type=int, default=5,
                       help='Number of persistence landscapes (default: 5)')
    parser.add_argument('--landscape-resolution', type=int, default=100,
                       help='Landscape resolution (default: 100)')
    parser.add_argument('--use-mean-landscape', action='store_true',
                       help='Use mean of all landscapes (reduces features)')
    parser.add_argument('--chunk-size', type=int, default=10,
                       help='Number of parquet files per chunk (default: 10)')
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel jobs (default: 1)')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum files to process (default: all)')
    
    args = parser.parse_args()
    
    # Validate paths
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("TOPOLOGICAL FEATURE EXTRACTION")
    print("=" * 80)
    print(f"\n📂 Input:  {data_dir}")
    print(f"📂 Output: {output_dir}")
    print(f"\n⚙️  Configuration:")
    print(f"   Embedding dimension: {args.embedding_dim}")
    print(f"   Embedding delay: {args.embedding_delay}")
    print(f"   N landscapes: {args.n_landscapes}")
    print(f"   Landscape resolution: {args.landscape_resolution}")
    print(f"   Use mean landscape: {args.use_mean_landscape}")
    print(f"   Chunk size: {args.chunk_size} files")
    print(f"   Parallel jobs: {args.n_jobs}")
    
    if args.use_mean_landscape:
        n_features_per_dim = args.landscape_resolution
        print(f"\n📊 Feature count per homology dimension: {n_features_per_dim}")
        print(f"   Total features (H0 + H1): {2 * n_features_per_dim}")
    else:
        n_features_per_dim = args.n_landscapes * args.landscape_resolution
        print(f"\n📊 Feature count per homology dimension: {n_features_per_dim}")
        print(f"   Total features (H0 + H1): {2 * n_features_per_dim}")
    
    print("\n" + "=" * 80)
    
    # Initialize extractor
    extractor = TopologicalFeatureExtractor(
        embedding_dim=args.embedding_dim,
        embedding_delay=args.embedding_delay,
        homology_dims=(0, 1),
        n_landscapes=args.n_landscapes,
        landscape_resolution=args.landscape_resolution,
        normalize=True,
        n_perm=500,  # Speed optimization
        use_mean_landscape=args.use_mean_landscape
    )
    
    # Process data
    try:
        extractor.process_chunk_by_chunk(
            input_path=data_dir,
            output_path=output_dir,
            chunk_size=args.chunk_size,
            max_files=args.max_files,
            n_jobs=args.n_jobs
        )
        
        print("\n" + "=" * 80)
        print("✅ TOPOLOGICAL FEATURE EXTRACTION COMPLETE")
        print("=" * 80)
        print(f"\n📂 Output: {output_dir}")
        print(f"   - topo_features.parquet")
        print(f"   - feature_names.txt")
        
    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
