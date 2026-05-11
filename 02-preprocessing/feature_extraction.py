"""
Chunked TSFresh Feature Extraction (File-based)

Optimized for stability and immediate feedback.
Process: Reads one parquet file -> Extracts features -> Saves chunk -> Repeats.
Final step: Merges all chunks into the final dataset.

Usage:
    python feature_extraction.py \
        --input ../data/raw/dataset \
        --output ../data/features/dataset/allfeatures \
        --feature-set efficient \
        --n-jobs 100
"""

import argparse
import warnings
from pathlib import Path
from typing import List, Optional
import time
import gc
import os
import shutil

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import (
    MinimalFCParameters,
    EfficientFCParameters,
    ComprehensiveFCParameters,
)

# Suppress warnings
warnings.filterwarnings("ignore")

REQUIRED_COLUMNS = ["series_id", "time", "data", "is_stationary", "primary_category", "sub_category"]

class ChunkedFeatureExtractor:
    def __init__(self, feature_set: str = "efficient", n_jobs: int = -1):
        self.feature_set = feature_set.lower()
        self.n_jobs = n_jobs if n_jobs and n_jobs > 0 else os.cpu_count()
        self.settings = self._select_feature_set(self.feature_set)
        
        print(f"Initialized with {self.n_jobs} cores.")
        print(f"Feature Set: {self.feature_set}")

    @staticmethod
    def _select_feature_set(feature_set: str):
        if feature_set == "minimal":
            return MinimalFCParameters()
        elif feature_set == "efficient":
            return EfficientFCParameters()
        elif feature_set == "comprehensive":
            return ComprehensiveFCParameters()
        else:
            raise ValueError(f"Unknown feature set: {feature_set}")

    def process_single_file(self, file_path: Path, temp_dir: Path, file_index: int, total_files: int):
        """Reads, processes, and saves a single parquet file."""
        
        # Show full path for absolute clarity
        display_name = str(file_path)
        
        start_time = time.time()
        print(f"\n[{file_index}/{total_files}] Processing: {display_name}")
        
        try:
            # 1. Read
            df = pd.read_parquet(file_path, columns=REQUIRED_COLUMNS)
            
            # 2. Clean
            initial_rows = len(df)
            df = df.dropna(subset=["data"])
            df = df[~df["data"].isin([np.inf, -np.inf])]
            
            if len(df) < initial_rows:
                print(f"   Cleaned {initial_rows - len(df)} bad rows.")
            
            if df.empty:
                print("   File is empty after cleaning. Skipping.")
                return

            # 3. Prepare
            labels = df.groupby("series_id")[["is_stationary", "primary_category", "sub_category"]].first()
            ts_data = df[["series_id", "time", "data"]]
            
            del df
            gc.collect()

            # 4. Extract
            features = extract_features(
                ts_data,
                column_id="series_id",
                column_sort="time",
                column_value="data",
                default_fc_parameters=self.settings,
                n_jobs=self.n_jobs,
                disable_progressbar=True, # Disable individual progress bars to reduce noise
                pivot=True
            )
            
            # 5. Impute (Per-chunk imputation is faster and safer for memory)
            impute(features)
            
            # 6. Save Chunk
            chunk_name = f"chunk_{file_index:04d}_{file_path.stem}.parquet"
            
            # Align labels
            features = features.sort_index()
            labels = labels.reindex(features.index)
            features.index.name = "series_id"
            labels.index.name = "series_id"
            
            feat_df = features.reset_index()
            lbl_df = labels.reset_index()
            
            # Ensure identifier column is always called series_id
            rename_map = {}
            if "index" in feat_df.columns:
                rename_map["index"] = "series_id"
            if "id" in feat_df.columns:
                rename_map["id"] = "series_id"
            if rename_map:
                feat_df = feat_df.rename(columns=rename_map)
            
            rename_lbl_map = {}
            if "index" in lbl_df.columns:
                rename_lbl_map["index"] = "series_id"
            if "id" in lbl_df.columns:
                rename_lbl_map["id"] = "series_id"
            if rename_lbl_map:
                lbl_df = lbl_df.rename(columns=rename_lbl_map)
            
            # Save features and labels separately in temp dir
            # Re-create temp directory if it was cleaned up externally
            temp_dir.mkdir(parents=True, exist_ok=True)
            feat_df.to_parquet(temp_dir / f"feat_{chunk_name}", index=False)
            lbl_df.to_parquet(temp_dir / f"lbl_{chunk_name}", index=False)
            
            duration = time.time() - start_time
            print(f"   Done in {duration:.1f}s. Shape: {features.shape}")
            
        except Exception as e:
            print(f"   Error processing {file_path.name}: {e}")
            # Don't stop the whole process, just log error
            
        finally:
            gc.collect()

    def merge_and_save(self, temp_dir: Path, output_path: Path):
        """Merges all chunks into the final output."""
        print("\nMerging all chunks...")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        feat_files = sorted(temp_dir.glob("feat_*.parquet"))
        lbl_files = sorted(temp_dir.glob("lbl_*.parquet"))
        
        if not feat_files:
            print("No features extracted. Exiting.")
            return

        # Read and concat
        print(f"   Combining {len(feat_files)} parts...")
        
        # Merge Features
        full_features = pd.concat([pd.read_parquet(f) for f in feat_files], ignore_index=True)
        full_features.to_parquet(output_path / "features.parquet", index=False)
        print(f"   Saved features.parquet {full_features.shape}")
        
        # Save feature names
        with open(output_path / "feature_names.txt", "w") as f:
            # Exclude series_id from feature names list
            cols = [c for c in full_features.columns if c != "series_id"]
            for col in cols:
                f.write(f"{col}\n")
        
        del full_features
        gc.collect()

        # Merge Labels
        full_labels = pd.concat([pd.read_parquet(f) for f in lbl_files], ignore_index=True)
        full_labels.to_parquet(output_path / "labels.parquet", index=False)
        print(f"   Saved labels.parquet {full_labels.shape}")

        print(f"\nProcess Complete! Output: {output_path}")
        
        # Cleanup temp
        print("Cleaning up temp files...")
        shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser(description="Chunked TSFresh Feature Extraction")
    parser.add_argument("--input", type=str, required=True, help="Input folder")
    parser.add_argument("--output", type=str, required=True, help="Output folder")
    parser.add_argument("--feature-set", type=str, default="efficient", choices=["minimal", "efficient", "comprehensive"])
    parser.add_argument("--n-jobs", type=int, default=0, help="Number of parallel jobs")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    temp_dir = output_path / "temp_chunks"
    
    # Create temp dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect files
    files = sorted(input_path.rglob("*.parquet"))
    print(f"Found {len(files)} files to process.")
    
    extractor = ChunkedFeatureExtractor(feature_set=args.feature_set, n_jobs=args.n_jobs)
    
    # Process Loop
    for i, file_path in enumerate(files, 1):
        extractor.process_single_file(file_path, temp_dir, i, len(files))
        
    # Final Merge
    extractor.merge_and_save(temp_dir, output_path)

if __name__ == "__main__":
    main()
