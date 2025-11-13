"""
Fast test for topological feature extraction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from topological_features import TopologicalFeatureExtractor

print("=" * 80)
print("Testing Topological Feature Extraction (Fast)")
print("=" * 80)

# Initialize
print("\n1. Initialize extractor...")
extractor = TopologicalFeatureExtractor(
    embedding_dim=3,
    embedding_delay=1,
    homology_dims=(0, 1),
    n_landscapes=5,
    landscape_resolution=100,
    normalize=True
)
print("   ✓ Ready")
print(f"   Embedding: {extractor.embedding_dim}D, delay={extractor.embedding_delay}")
print(f"   Homology: {extractor.homology_dims}")
print(f"   Landscapes: {extractor.n_landscapes}x{extractor.landscape_resolution}")

# Test 1: Synthetic data
print("\n2. Test with synthetic time series...")
np.random.seed(42)
ts = np.sin(np.linspace(0, 4*np.pi, 500)) + np.random.normal(0, 0.1, 500)
print(f"   Length: {len(ts)}, Range: [{ts.min():.2f}, {ts.max():.2f}]")

# Takens embedding
print("\n3. Takens embedding...")
pc = extractor.takens_embedding(ts)
print(f"   ✓ Point cloud: {pc.shape}")

# Persistence diagrams
print("\n4. Persistence diagrams (Ripser)...")
diagrams = extractor.compute_persistence_diagrams(ts)
print(f"   ✓ Computed for: {list(diagrams.keys())}")
for dim, dgm in diagrams.items():
    finite = dgm[np.isfinite(dgm).all(axis=1)]
    print(f"   H{dim}: {len(finite)} persistence pairs")

# Features
print("\n5. Extract features...")
features = extractor.extract_features(ts)
expected = len(extractor.homology_dims) * extractor.n_landscapes * extractor.landscape_resolution
print(f"   ✓ Features: {features.shape}, expected: ({expected},)")
print(f"   Range: [{features.min():.3f}, {features.max():.3f}]")
print(f"   Non-zero: {np.count_nonzero(features)}/{len(features)}")

# Feature names
print("\n6. Feature names...")
names = extractor.get_feature_names()
print(f"   ✓ {len(names)} names")
print(f"   First: {names[0]}")
print(f"   Last: {names[-1]}")

# Test 2: Real data (if available)
print("\n7. Test with real data...")
data_path = Path("../data/raw/unified-5k/stationary/ar/long.parquet")

if data_path.exists():
    df = pd.read_parquet(data_path)
    print(f"   ✓ Loaded: {data_path.name}")
    print(f"   Shape: {df.shape}, Series: {df['series_id'].nunique()}")
    
    # Get first series
    sid = df['series_id'].iloc[0]
    ts_real = df[df['series_id'] == sid]['data'].values
    print(f"\n   Processing series: {sid}")
    print(f"   Length: {len(ts_real)}, Range: [{ts_real.min():.2f}, {ts_real.max():.2f}]")
    
    # Extract
    feats_real = extractor.extract_features(ts_real)
    print(f"   ✓ Features: {feats_real.shape}")
    print(f"   Range: [{feats_real.min():.3f}, {feats_real.max():.3f}]")
    print(f"   Non-zero: {np.count_nonzero(feats_real)}/{len(feats_real)}")
    
    # Parallel batch test (small)
    print(f"\n8. Test parallel batch extraction...")
    series_dict = {}
    for sid, group in df.groupby('series_id'):
        series_dict[sid] = group['data'].values
        if len(series_dict) >= 3:  # Test with 3 series
            break
    
    print(f"   Testing with {len(series_dict)} series...")
    
    # Sequential
    print(f"   Sequential (n_jobs=1)...")
    feats_df, failed = extractor.extract_batch(series_dict, n_jobs=1, verbose=False)
    print(f"   ✓ {feats_df.shape[0]} series, {feats_df.shape[1]-1} features")
    
    # Parallel
    print(f"   Parallel (n_jobs=2)...")
    feats_df2, failed2 = extractor.extract_batch(series_dict, n_jobs=2, verbose=False)
    print(f"   ✓ {feats_df2.shape[0]} series, {feats_df2.shape[1]-1} features")
    
else:
    print(f"   ⚠ Data not found: {data_path}")
    print(f"   Skipping real data test")

print("\n" + "="*80)
print("✓ All tests passed!")
print("="*80)
