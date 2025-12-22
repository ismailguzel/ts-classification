#!/bin/bash
# Preprocessing Pipeline for 20K Dataset
# Steps: Feature Extraction -> Leakage Removal -> Feature Selection

set -e  # Exit on error

# Load required module (if on cluster)
if command -v module &> /dev/null; then
    module load apps/truba-ai/gpu-2024.0
fi

# Activate environment
conda activate ts-sktime

# --- Configuration ---
SCALE="20k"
N_JOBS=100
FEATURE_SET="efficient"
TOP_K_FEATURES=100

# Paths
BASE_DIR=$(pwd)
DATA_RAW="$BASE_DIR/data/raw/unified-$SCALE"
DATA_FEATURES_STAT="$BASE_DIR/data/features/unified-$SCALE/statistical"
DATA_FEATURES_TOPO="$BASE_DIR/data/features/unified-$SCALE/topological"
DATA_FEATURES_COMBINED="$BASE_DIR/data/features/unified-$SCALE/combined"
DATA_SELECTED_STAT="$BASE_DIR/data/features/unified-$SCALE/statistical_selected"
DATA_SELECTED_TOPO="$BASE_DIR/data/features/unified-$SCALE/topological_selected"
DATA_SELECTED_COMBINED="$BASE_DIR/data/features/unified-$SCALE/combined_selected"

# Validate Input
if [ ! -d "$DATA_RAW" ]; then
    echo "✗ Error: Input directory not found: $DATA_RAW"
    echo "  Please run 'run-generation.sh' first."
    exit 1
fi

# Create Output Directories
mkdir -p "$DATA_FEATURES_STAT"
mkdir -p "$DATA_FEATURES_TOPO"
mkdir -p "$DATA_FEATURES_COMBINED"
mkdir -p "$DATA_SELECTED_STAT"
mkdir -p "$DATA_SELECTED_TOPO"
mkdir -p "$DATA_SELECTED_COMBINED"

echo "============================================================"
echo "Starting Preprocessing Pipeline ($SCALE)"
echo "============================================================"
echo "Input:  $DATA_RAW"
echo "Output: "
echo "  - Statistical: $DATA_SELECTED_STAT"
echo "  - Topological: $DATA_SELECTED_TOPO"
echo "  - Combined: $DATA_SELECTED_COMBINED"
echo ""

# 1. Statistical Feature Extraction (TSFresh)
echo "============================================================"
echo "Step 1: Statistical Feature Extraction (TSFresh)"
echo "============================================================"
cd 02-preprocessing
python feature_extraction.py \
    --input "$DATA_RAW" \
    --output "$DATA_FEATURES_STAT" \
    --feature-set $FEATURE_SET \
    --n-jobs $N_JOBS

if [ ! -f "$DATA_FEATURES_STAT/features.parquet" ]; then
    echo "✗ Error: Statistical feature extraction failed."
    exit 1
fi

# 2. Leakage Removal (Statistical)
echo ""
echo "============================================================"
echo "Step 2: Removing Data Leakage Features (Statistical)"
echo "============================================================"
python remove_leakage_features.py \
    --input "$DATA_FEATURES_STAT" \
    --no-backup

# 3. Statistical Feature Selection
echo ""
echo "============================================================"
echo "Step 3: Selecting Top $TOP_K_FEATURES Statistical Features"
echo "============================================================"
python feature_selection.py \
    --input "$DATA_FEATURES_STAT" \
    --output "$DATA_SELECTED_STAT" \
    --n-features $TOP_K_FEATURES \
    --method mutual_info \
    --target all

if [ ! -d "$DATA_SELECTED_STAT" ] || [ -z "$(ls -A $DATA_SELECTED_STAT)" ]; then
    echo "✗ Error: Statistical feature selection failed."
    exit 1
fi

# 4. Topological Feature Extraction
echo ""
echo "============================================================"
echo "Step 4: Topological Feature Extraction (Persistent Homology)"
echo "============================================================"
echo "⚠️  Switching to ts-top conda environment for topological features..."
conda deactivate
conda activate ts-top

python extract_topo_features.py \
    --data-dir "$DATA_RAW" \
    --output-dir "$DATA_FEATURES_TOPO" \
    --use-mean-landscape \
    --n-jobs $N_JOBS \
    --chunk-size 10

if [ ! -f "$DATA_FEATURES_TOPO/features.parquet" ]; then
    echo "✗ Error: Topological feature extraction failed."
    exit 1
fi

echo "✓ Switching back to ts-sktime environment..."
conda deactivate
conda activate ts-sktime

# 5. Topological Feature Selection
echo ""
echo "============================================================"
echo "Step 5: Selecting Top 50 Topological Features"
echo "============================================================"
python feature_selection.py \
    --input "$DATA_FEATURES_TOPO" \
    --output "$DATA_SELECTED_TOPO" \
    --n-features 50 \
    --method mutual_info \
    --target all

if [ ! -d "$DATA_SELECTED_TOPO" ] || [ -z "$(ls -A $DATA_SELECTED_TOPO)" ]; then
    echo "✗ Error: Topological feature selection failed."
    exit 1
fi

# 6. Combine Statistical and Topological Features
echo ""
echo "============================================================"
echo "Step 6: Combining Statistical + Topological Features"
echo "============================================================"
python combine_features.py \
    --stat-dir "$DATA_FEATURES_STAT" \
    --topo-dir "$DATA_FEATURES_TOPO" \
    --output-dir "$DATA_FEATURES_COMBINED"

if [ ! -f "$DATA_FEATURES_COMBINED/features.parquet" ]; then
    echo "✗ Error: Feature combination failed."
    exit 1
fi

# 7. Combined Feature Selection
echo ""
echo "============================================================"
echo "Step 7: Selecting Top 150 Combined Features"
echo "============================================================"
python feature_selection.py \
    --input "$DATA_FEATURES_COMBINED" \
    --output "$DATA_SELECTED_COMBINED" \
    --n-features 150 \
    --method mutual_info \
    --target all

if [ ! -d "$DATA_SELECTED_COMBINED" ] || [ -z "$(ls -A $DATA_SELECTED_COMBINED)" ]; then
    echo "✗ Error: Combined feature selection failed."
    exit 1
fi

echo ""
echo "============================================================"
echo "Preprocessing Complete!"
echo "============================================================"
echo "Statistical Features: $DATA_SELECTED_STAT (100 features)"
echo "Topological Features: $DATA_SELECTED_TOPO (50 features)"
echo "Combined Features: $DATA_SELECTED_COMBINED (150 features)"
echo "============================================================"
