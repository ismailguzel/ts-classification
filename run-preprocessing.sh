#!/bin/bash
# Preprocessing Pipeline
#
# Usage:
#   bash run-preprocessing.sh                        # full, TSFresh only
#   bash run-preprocessing.sh test                   # test dataset, TSFresh only
#   bash run-preprocessing.sh topo-test              # topo-test (10 classes), TSFresh only
#   bash run-preprocessing.sh test topo              # test, topology (sublevel+h1, default)
#   bash run-preprocessing.sh topo-test topo         # topo-test, topology
#   bash run-preprocessing.sh test topo sublevel     # sublevel only (no Ripser)
#   bash run-preprocessing.sh test topo takens       # takens H0+H1 (Ripser)
#   bash run-preprocessing.sh test topo both         # all diagrams

set -e

PYTHON="${PYTHON:-python}"

MODE=${1:-full}          # full | test | topo-test
FEATURES=${2:-}          # topo  (leave empty for TSFresh only)
TOPO_METHOD=${3:-sublevel+h1}   # sublevel+h1 | sublevel | takens | both

N_JOBS=-1
FEATURE_SET="efficient"
TOP_K_FEATURES=100
BASE_DIR=$(pwd)

if [ "$MODE" == "test" ]; then
    DATA_RAW="$BASE_DIR/data/raw/test"
    DATA_FEATURES="$BASE_DIR/data/features/test/allfeatures"
    DATA_SELECTED="$BASE_DIR/data/features/test/selected"
    DATA_TOPO="$BASE_DIR/data/features/test/topological"
    DATA_TOPO_SEL="$BASE_DIR/data/features/test/topological_selected"
elif [ "$MODE" == "topo-test" ]; then
    DATA_RAW="$BASE_DIR/data/raw/topo-test"
    DATA_FEATURES="$BASE_DIR/data/features/topo-test/allfeatures"
    DATA_SELECTED="$BASE_DIR/data/features/topo-test/selected"
    DATA_TOPO="$BASE_DIR/data/features/topo-test/topological"
    DATA_TOPO_SEL="$BASE_DIR/data/features/topo-test/topological_selected"
else
    DATA_RAW="$BASE_DIR/data/raw/dataset"
    DATA_FEATURES="$BASE_DIR/data/features/dataset/allfeatures"
    DATA_SELECTED="$BASE_DIR/data/features/dataset/selected"
    DATA_TOPO="$BASE_DIR/data/features/dataset/topological"
    DATA_TOPO_SEL="$BASE_DIR/data/features/dataset/topological_selected"
fi

if [ ! -d "$DATA_RAW" ] || [ -z "$(ls -A $DATA_RAW 2>/dev/null)" ]; then
    echo "Error: Input not found: $DATA_RAW"
    echo "  Run 'bash run-generation.sh $MODE' first."
    exit 1
fi

echo "============================================================"
echo "Preprocessing — $MODE  |  features: ${FEATURES:-tsfresh}"
echo "============================================================"
echo "Input : $DATA_RAW"
echo "Start : $(date)"

cd 02-preprocessing

# ----------------------------------------------------------
# TSFresh pipeline  (tsfresh veya hybrid)
# ----------------------------------------------------------

if [ "$FEATURES" != "topo" ]; then
    echo ""
    echo "--- TSFresh ---"
    echo "Output : $DATA_SELECTED"

    mkdir -p "$DATA_FEATURES" "$DATA_SELECTED"

    echo "Step 1: Feature Extraction (TSFresh $FEATURE_SET)"
    $PYTHON feature_extraction.py \
        --input   "$DATA_RAW" \
        --output  "$DATA_FEATURES" \
        --feature-set $FEATURE_SET \
        --n-jobs  $N_JOBS

    [ -f "$DATA_FEATURES/features.parquet" ] || { echo "Error: Feature extraction failed."; exit 1; }

    echo "Step 2: Leakage Removal"
    $PYTHON remove_leakage_features.py \
        --input "$DATA_FEATURES" \
        --no-backup

    echo "Step 3: Feature Selection (top $TOP_K_FEATURES, train-only MI)"
    $PYTHON feature_selection.py \
        --input      "$DATA_FEATURES" \
        --output     "$DATA_SELECTED" \
        --n-features $TOP_K_FEATURES \
        --method     mutual_info \
        --target     primary

    [ -d "$DATA_SELECTED/primary" ] || { echo "Error: Feature selection failed."; exit 1; }
fi

# ----------------------------------------------------------
# Topology pipeline  (topo veya hybrid)
# ----------------------------------------------------------

if [ "$FEATURES" == "topo" ] || [ "$FEATURES" == "hybrid" ]; then
    echo ""
    echo "--- Topology ---"
    echo "Output : $DATA_TOPO_SEL"

    mkdir -p "$DATA_TOPO" "$DATA_TOPO_SEL"

    echo "Step 1: Topological Feature Extraction (method=$TOPO_METHOD)"
    $PYTHON topology_extraction.py \
        --input      "$DATA_RAW" \
        --output     "$DATA_TOPO" \
        --method     "$TOPO_METHOD" \
        --auto-delay \
        --auto-dim   \
        --n-perm     500 \
        --n-jobs     $N_JOBS

    [ -f "$DATA_TOPO/features.parquet" ] || { echo "Error: Topology extraction failed."; exit 1; }

    echo "Step 2: Topology Feature Selection (top $TOP_K_FEATURES, train-only MI)"
    $PYTHON feature_selection.py \
        --input      "$DATA_TOPO" \
        --output     "$DATA_TOPO_SEL" \
        --n-features $TOP_K_FEATURES \
        --method     mutual_info \
        --target     primary

    [ -d "$DATA_TOPO_SEL/primary" ] || { echo "Error: Topology feature selection failed."; exit 1; }
fi

cd "$BASE_DIR"

echo ""
echo "============================================================"
echo "Preprocessing Complete!  features=${FEATURES:-tsfresh}"
if [ "$FEATURES" != "topo" ]; then
    echo "  TSFresh  : $DATA_SELECTED"
fi
if [ "$FEATURES" == "topo" ] || [ "$FEATURES" == "hybrid" ]; then
    echo "  Topology : $DATA_TOPO_SEL"
fi
echo "End: $(date)"
echo "============================================================"
