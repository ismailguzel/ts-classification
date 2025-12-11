"""
Remove data leakage features (stationary tests) from extracted features.

This script removes features like Augmented Dickey-Fuller test results
that directly measure stationarity, which would cause data leakage when
predicting is_stationary.

Usage:
    python remove_leakage_features.py --input ../data/features/unified-5k
"""

import argparse
from pathlib import Path
import pandas as pd
import shutil
from datetime import datetime

# Features that cause data leakage
# These are statistical tests that directly measure stationarity
LEAKAGE_PATTERNS = [
    # Augmented Dickey-Fuller (ADF) test
    'augmented_dickey_fuller',
    'dickey_fuller',
    'adf_test',
    'adf',
    
    # KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test
    'kpss',
    'kpss_test',
    'kwiatkowski',
    
    # Phillips-Perron (PP) test
    'phillips_perron',
    'phillips_peron',  # typo variant
    'pp_test',
    'perron',
    
    # Unit root tests (general)
    'unit_root',
    'unitroot',
    'unit_root_test',
    
    # Explicit stationarity tests
    'stationarity_test',
    'stationary_test',
    'is_stationary',
    'test_stationarity',
    
    # Variance ratio test (Lo-MacKinlay)
    'variance_ratio_test',
    'lo_mackinlay',
    
    # Other potential leakage
    'zivot_andrews',  # Structural break test
    'breakpoint_test',
    'cointegration_test',  # If present
]


def find_leakage_features(feature_cols):
    """Find features matching leakage patterns."""
    leakage_features = []
    for col in feature_cols:
        col_lower = col.lower()
        for pattern in LEAKAGE_PATTERNS:
            if pattern in col_lower:
                leakage_features.append(col)
                break
    return leakage_features


def clean_feature_file(file_path, backup=True, dry_run=False):
    """
    Remove leakage features from a parquet file.
    
    Args:
        file_path: Path to feature parquet file
        backup: Create backup before modifying
        dry_run: Only show what would be done
    """
    try:
        # Read file
        df = pd.read_parquet(file_path)
        
        # Identify columns
        meta_cols = ['series_id', 'is_stationary', 'primary_category', 'sub_category']
        feature_cols = [col for col in df.columns if col not in meta_cols]
        
        # Find leakage features
        leakage_features = find_leakage_features(feature_cols)
        
        if not leakage_features:
            return 0, 0
        
        print(f"\n📁 {file_path.name}")
        print(f"   Total features: {len(feature_cols)}")
        print(f"   Leakage features: {len(leakage_features)}")
        for feat in leakage_features:
            print(f"      - {feat}")
        
        if dry_run:
            print(f"   [DRY RUN] Would remove {len(leakage_features)} features")
            return len(leakage_features), len(feature_cols)
        
            # Backup
            if backup:
                backup_path = file_path.with_suffix('.parquet.backup')
                shutil.copy2(file_path, backup_path)
                print(f"   Backup created: {backup_path.name}")
        
        # Remove leakage features
        clean_cols = [col for col in df.columns if col not in leakage_features]
        df_clean = df[clean_cols]
        
        # Save
        df_clean.to_parquet(file_path, index=False)
        print(f"   Cleaned: {len(feature_cols)} -> {len(clean_cols) - len(meta_cols)} features")
        
        return len(leakage_features), len(feature_cols)
        
    except Exception as e:
        print(f"   Error: {e}")
        return 0, 0


def update_feature_names_file(txt_file_path, leakage_features):
    """Update feature_names.txt file to remove leakage features."""
    if not txt_file_path.exists():
        return
    
    try:
        with open(txt_file_path, 'r') as f:
            all_features = [line.strip() for line in f if line.strip()]
        
        # Remove leakage features
        clean_features = [f for f in all_features if f not in leakage_features]
        
        if len(clean_features) < len(all_features):
            # Backup
            backup_path = txt_file_path.with_suffix('.txt.backup')
            shutil.copy2(txt_file_path, backup_path)
            
            # Save cleaned
            with open(txt_file_path, 'w') as f:
                for feat in clean_features:
                    f.write(f"{feat}\n")
            
            print(f"   Updated: {txt_file_path.name}")
            print(f"     {len(all_features)} -> {len(clean_features)} features")
    
    except Exception as e:
        print(f"   Error updating {txt_file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove data leakage features from TSFresh outputs"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../data/features/unified-5k",
        help="Input directory containing features"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Input path not found: {input_path}")
        return
    
    print("="*80)
    print("REMOVE DATA LEAKAGE FEATURES")
    print("="*80)
    print(f"Input: {input_path}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Backup: {'No' if args.no_backup else 'Yes'}")
    print(f"\nChecking for {len(LEAKAGE_PATTERNS)} leakage patterns:")
    print("  • ADF (Augmented Dickey-Fuller)")
    print("  • KPSS (Kwiatkowski-Phillips-Schmidt-Shin)")
    print("  • Phillips-Perron (PP)")
    print("  • Unit Root tests")
    print("  • Stationarity tests")
    print("  • Variance Ratio tests")
    print("="*80)
    
    # Find all feature files
    feature_files = list(input_path.glob("**/features*.parquet"))
    feature_files = [f for f in feature_files if '.backup' not in str(f)]
    
    print(f"\nFound {len(feature_files)} feature files")
    
    total_removed = 0
    total_features = 0
    cleaned_files = 0
    
    for file_path in sorted(feature_files):
        removed, total = clean_feature_file(
            file_path,
            backup=not args.no_backup,
            dry_run=args.dry_run
        )
        
        if removed > 0:
            cleaned_files += 1
            total_removed += removed
            total_features += total
            
            # Update corresponding feature_names.txt
            if not args.dry_run:
                # Try to find feature_names.txt in same directory
                txt_file = file_path.parent / "feature_names.txt"
                if not txt_file.exists():
                    # Try alternative name
                    base_name = file_path.stem.replace('features', 'feature_names')
                    txt_file = file_path.parent / f"{base_name}.txt"
                
                if txt_file.exists():
                    # Read leakage features from current file
                    df = pd.read_parquet(file_path)
                    feature_cols = [col for col in df.columns 
                                   if col not in ['series_id', 'is_stationary', 
                                                 'primary_category', 'sub_category']]
                    leakage_in_txt = find_leakage_features(
                        [line.strip() for line in open(txt_file)]
                    )
                    if leakage_in_txt:
                        update_feature_names_file(txt_file, leakage_in_txt)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Files processed: {len(feature_files)}")
    print(f"Files cleaned: {cleaned_files}")
    print(f"Total leakage features removed: {total_removed}")
    
    if args.dry_run:
        print(f"\nDRY RUN - No files were modified")
        print(f"Run without --dry-run to apply changes")
    else:
        print(f"\nCleanup completed!")
        if not args.no_backup:
            print(f"   Backup files created with .backup extension")
    
    print("="*80)


if __name__ == "__main__":
    main()

