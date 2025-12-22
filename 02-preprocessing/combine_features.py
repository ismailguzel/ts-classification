"""
Combine Statistical and Topological Features

Merges TSFresh features with topological features for enhanced classification.

Usage:
    python combine_features.py \
        --stat-dir ../data/features/unified-20k/statistical \
        --topo-dir ../data/features/unified-20k/topological \
        --output-dir ../data/features/unified-20k/combined
"""

import argparse
from pathlib import Path
import pandas as pd
import sys


def combine_features(stat_path, topo_path, output_path, verbose=True):
    """
    Combine statistical and topological features.
    
    Args:
        stat_path: Path to statistical features directory
        topo_path: Path to topological features directory
        output_path: Output directory for combined features
        verbose: Print progress messages
    """
    stat_path = Path(stat_path)
    topo_path = Path(topo_path)
    output_path = Path(output_path)
    
    if verbose:
        print("=" * 80)
        print("COMBINING STATISTICAL AND TOPOLOGICAL FEATURES")
        print("=" * 80)
        print(f"\n📂 Statistical: {stat_path}")
        print(f"📂 Topological: {topo_path}")
        print(f"📂 Output: {output_path}")
    
    # Validate inputs
    stat_features_file = stat_path / 'features.parquet'
    stat_labels_file = stat_path / 'labels.parquet'
    topo_features_file = topo_path / 'features.parquet'
    topo_labels_file = topo_path / 'labels.parquet'
    
    if not stat_features_file.exists():
        print(f"❌ Error: Statistical features not found: {stat_features_file}")
        sys.exit(1)
    
    if not topo_features_file.exists():
        print(f"❌ Error: Topological features not found: {topo_features_file}")
        sys.exit(1)
    
    # Load features
    if verbose:
        print("\n📥 Loading features...")
    
    stat_features = pd.read_parquet(stat_features_file)
    topo_features = pd.read_parquet(topo_features_file)
    
    if verbose:
        print(f"   Statistical: {stat_features.shape}")
        print(f"   Topological: {topo_features.shape}")
    
    # Ensure series_id is in columns
    if stat_features.index.name == 'series_id':
        stat_features = stat_features.reset_index()
    if topo_features.index.name == 'series_id':
        topo_features = topo_features.reset_index()
    
    # Merge on series_id
    if verbose:
        print("\n🔗 Merging features...")
    
    combined = stat_features.merge(
        topo_features, 
        on='series_id', 
        how='inner',
        suffixes=('_stat', '_topo')
    )
    
    if verbose:
        print(f"   Combined: {combined.shape}")
        print(f"   Series matched: {len(combined)}")
        
        n_stat = len([c for c in stat_features.columns if c != 'series_id'])
        n_topo = len([c for c in topo_features.columns if c != 'series_id'])
        n_combined = len([c for c in combined.columns if c != 'series_id'])
        
        print(f"\n📊 Feature counts:")
        print(f"   Statistical: {n_stat}")
        print(f"   Topological: {n_topo}")
        print(f"   Combined: {n_combined} (expected: {n_stat + n_topo})")
        
        if n_combined != n_stat + n_topo:
            print(f"   ⚠️  Warning: Feature count mismatch!")
    
    # Load labels (use statistical labels as reference)
    if verbose:
        print("\n📥 Loading labels...")
    
    labels = pd.read_parquet(stat_labels_file)
    if verbose:
        print(f"   Labels: {labels.shape}")
    
    # Save combined features
    output_path.mkdir(parents=True, exist_ok=True)
    
    combined_features_file = output_path / 'features.parquet'
    combined_labels_file = output_path / 'labels.parquet'
    
    if verbose:
        print(f"\n💾 Saving combined features...")
        print(f"   {combined_features_file}")
    
    combined.to_parquet(combined_features_file, index=False)
    labels.to_parquet(combined_labels_file, index=False)
    
    # Save feature names
    feature_names_file = output_path / 'feature_names.txt'
    feature_names = [c for c in combined.columns if c != 'series_id']
    
    with open(feature_names_file, 'w') as f:
        f.write('\n'.join(feature_names))
    
    if verbose:
        print(f"   {feature_names_file}")
        print(f"\n✅ COMBINATION COMPLETE")
        print("=" * 80)
        print(f"\n📊 Summary:")
        print(f"   Total series: {len(combined)}")
        print(f"   Total features: {len(feature_names)}")
        print(f"   Statistical contribution: {n_stat} ({n_stat/len(feature_names)*100:.1f}%)")
        print(f"   Topological contribution: {n_topo} ({n_topo/len(feature_names)*100:.1f}%)")
    
    return combined, labels


def main():
    parser = argparse.ArgumentParser(description='Combine statistical and topological features')
    parser.add_argument('--stat-dir', type=str, required=True,
                       help='Statistical features directory')
    parser.add_argument('--topo-dir', type=str, required=True,
                       help='Topological features directory')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for combined features')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output messages')
    
    args = parser.parse_args()
    
    try:
        combine_features(
            stat_path=args.stat_dir,
            topo_path=args.topo_dir,
            output_path=args.output_dir,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
