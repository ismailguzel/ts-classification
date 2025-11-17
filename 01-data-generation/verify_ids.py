#!/usr/bin/env python3
"""
Verify global uniqueness of series_id across a generated dataset.

Usage:
  python verify_ids.py --data-path ../data/raw/unified-test
  python verify_ids.py --data-path ../data/raw/unified-90k

This script scans all parquet files under the given path and checks:
  - Total number of parquet files
  - Total number of series entries (sum of per-file unique series ids)
  - Number of globally unique series ids
  - Any overlaps (ids that appear in more than one file)
  - Basic range information for sanity (min/max id)

Exit code:
  - 0 if no overlaps detected
  - 1 if overlaps are found or an error occurs
"""

import argparse
from pathlib import Path
import pandas as pd
from collections import Counter


def verify_ids(data_path: Path, sample_duplicates: int = 10) -> int:
    parquet_files = sorted(data_path.glob("**/*.parquet"))
    if not parquet_files:
        print(f"No parquet files found under: {data_path}")
        return 1

    print(f"Scanning {len(parquet_files)} parquet files under: {data_path}")

    seen = set()
    dupes = Counter()
    per_file_counts = []

    for pf in parquet_files:
        try:
            ids = pd.read_parquet(pf, columns=["series_id"])['series_id'].astype(int)
        except Exception as e:
            print(f"    Failed to read series_id from {pf}: {e}")
            return 1
        unique_ids = set(ids.unique().tolist())
        per_file_counts.append(len(unique_ids))
        # overlaps
        overlap = unique_ids & seen
        for oid in overlap:
            dupes[oid] += 1
        seen.update(unique_ids)

    total_series_sum = sum(per_file_counts)
    unique_series = len(seen)
    overlap_count = len(dupes)

    print("\nSummary:")
    print(f"  Files scanned           : {len(parquet_files)}")
    print(f"  Sum per-file unique IDs : {total_series_sum:,}")
    print(f"  Global unique IDs       : {unique_series:,}")
    print(f"  Overlapping IDs         : {overlap_count:,}")

    if unique_series > 0:
        print(f"  Min ID                  : {min(seen)}")
        print(f"  Max ID                  : {max(seen)}")

    if overlap_count:
        print("\nExamples of overlapping IDs:")
        for oid, _ in list(dupes.items())[:sample_duplicates]:
            print(f"  - {oid}")
        print("\n ID collisions detected. Ensure start_id is propagated or regenerate.")
        return 1

    print("\n No overlaps found. series_id values are globally unique.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Verify global uniqueness of series_id across a dataset")
    parser.add_argument("--data-path", type=str, required=True, help="Path to dataset root (e.g., ../data/raw/unified-test)")
    args = parser.parse_args()

    exit(verify_ids(Path(args.data_path)))


if __name__ == "__main__":
    main()
