"""
Comprehensive NaN Inspector Tool for Time Series Parquet Files.

Modes:
1. SCAN: Scans a directory recursively to find files containing NaNs.
2. INSPECT: Analyzes a specific file and series_id to show where NaNs occur.

Usage:
    # 1. Scan for problems
    python inspect_nans.py --mode scan --input ../data/raw/unified-20k --output nan_report.csv

    # 2. Inspect a specific series (found in the scan report)
    python inspect_nans.py --mode inspect \
        --file ../data/raw/unified-20k/volatility/aparch/long.parquet \
        --series-id 15519
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
import sys

# --- SCAN MODE FUNCTIONS ---

def check_file(file_path):
    """Checks a single parquet file for NaNs."""
    try:
        # Read only necessary columns for speed
        df = pd.read_parquet(file_path, columns=["series_id", "time", "data"])
        
        # Check for NaNs in 'data' column
        nan_rows = df[df["data"].isna()]
        
        # Check for Infinite values
        inf_rows = df[np.isinf(df["data"])]
        
        if not nan_rows.empty or not inf_rows.empty:
            affected_series = pd.unique(
                pd.concat([nan_rows["series_id"], inf_rows["series_id"]])
            )
            
            return {
                "file": str(file_path),
                "nan_count": len(nan_rows),
                "inf_count": len(inf_rows),
                "affected_series_count": len(affected_series),
                "example_series_ids": list(affected_series[:5])
            }
    except Exception as e:
        return {"file": str(file_path), "error": str(e)}
    return None

def run_scan(input_path, output_file, n_jobs):
    """Scans directory for problematic files."""
    path = Path(input_path)
    if not path.exists():
        print(f"Error: Directory {path} not found.")
        return

    files = sorted(path.rglob("*.parquet"))
    print(f"Scanning {len(files)} files in {path}...")
    print(f"Using {n_jobs} parallel jobs.")

    results = Parallel(n_jobs=n_jobs)(delayed(check_file)(f) for f in files)
    problems = [r for r in results if r is not None]

    if problems:
        print(f"\nFound {len(problems)} files with issues!")
        df_problems = pd.DataFrame(problems)
        
        # Reorder columns for readability
        cols = ["file", "nan_count", "inf_count", "affected_series_count", "example_series_ids"]
        if "error" in df_problems.columns:
            cols.append("error")
        df_problems = df_problems[cols]
        
        print(df_problems.to_string(index=False))
        df_problems.to_csv(output_file, index=False)
        print(f"\nReport saved to {output_file}")
    else:
        print("\nNo NaNs or Infs found in any file.")


# --- INSPECT MODE FUNCTIONS ---

def run_inspect(file_path, target_id):
    """Analyzes a specific series in a specific file."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File {path} not found.")
        return

    print(f"Reading {path}...")
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return

    # Convert target_id to correct type if necessary
    # Usually series_id is int or string. Try to match.
    if df["series_id"].dtype == np.int64 or df["series_id"].dtype == np.int32:
        try:
            target_id = int(target_id)
        except ValueError:
            pass # Keep as string if conversion fails
            
    series_data = df[df["series_id"] == target_id]

    if series_data.empty:
        print(f"Series ID {target_id} not found in this file.")
        print(f"Available IDs (first 5): {df['series_id'].unique()[:5]}")
        return

    print(f"\nAnalysis for Series ID: {target_id}")
    print("-" * 50)
    print(f"Total Length: {len(series_data)}")
    
    # Check for NaNs
    nans = series_data[series_data["data"].isna()]
    print(f"NaN Count   : {len(nans)}")
    
    # Check for Infs
    infs = series_data[np.isinf(series_data["data"])]
    print(f"Inf Count   : {len(infs)}")

    if not nans.empty:
        print("\nNaN Details:")
        print(f"   First NaN at time: {nans['time'].min()}")
        print(f"   Last NaN at time : {nans['time'].max()}")
        print("\n   First 5 NaN rows:")
        print(nans[["time", "data"]].head().to_string(index=False))

    if not infs.empty:
        print("\nInf Details:")
        print(f"   First Inf at time: {infs['time'].min()}")
        print(f"   Last Inf at time : {infs['time'].max()}")
        print("\n   First 5 Inf rows:")
        print(infs[["time", "data"]].head().to_string(index=False))

    # Valid Data Stats
    valid_data = series_data[~series_data["data"].isna() & ~np.isinf(series_data["data"])]
    if not valid_data.empty:
        print("\nValid Data Statistics:")
        print(f"   Count: {len(valid_data)}")
        print(f"   Mean : {valid_data['data'].mean():.4f}")
        print(f"   Min  : {valid_data['data'].min():.4f}")
        print(f"   Max  : {valid_data['data'].max():.4f}")
        
        # Show snippet of valid data
        print("\n   Valid Data Snippet:")
        print(valid_data[["time", "data"]].head().to_string(index=False))
    else:
        print("\nThis series contains NO valid data!")


# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Inspect and Detect NaNs in Parquet Time Series")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Operation mode")

    # SCAN Parser
    scan_parser = subparsers.add_parser("scan", help="Scan directory for NaNs")
    scan_parser.add_argument("--input", type=str, required=True, help="Input directory to scan")
    scan_parser.add_argument("--output", type=str, default="nan_report.csv", help="Output CSV file")
    scan_parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs")

    # INSPECT Parser
    inspect_parser = subparsers.add_parser("inspect", help="Inspect specific file and series")
    inspect_parser.add_argument("--file", type=str, required=True, help="Path to parquet file")
    inspect_parser.add_argument("--series-id", type=str, required=True, help="Series ID to inspect")

    args = parser.parse_args()

    if args.mode == "scan":
        run_scan(args.input, args.output, args.n_jobs)
    elif args.mode == "inspect":
        run_inspect(args.file, args.series_id)

if __name__ == "__main__":
    main()
