#!/bin/bash
# Training Pipeline — Flat classifier
#
# Usage:
#   bash run-training.sh                       # full dataset, TSFresh features
#   bash run-training.sh test                  # test dataset (39 classes), TSFresh
#   bash run-training.sh topo-test             # topo-test dataset (10 classes), TSFresh
#   bash run-training.sh test topo             # test, topology features only
#   bash run-training.sh topo-test topo        # topo-test, topology features only
#   bash run-training.sh test hybrid           # test, TSFresh + topology merged
#   bash run-training.sh topo-test hybrid      # topo-test, TSFresh + topology merged

set -e

PYTHON="${PYTHON:-python}"

MODE=${1:-full}         # full | test | topo-test
FEATURES=${2:-tsfresh}  # tsfresh | topo | hybrid

BASE_DIR=$(pwd)

if [ "$MODE" == "test" ]; then
    DATA_TSFRESH="$BASE_DIR/data/features/test/selected"
    DATA_TOPO="$BASE_DIR/data/features/test/topological_selected"
elif [ "$MODE" == "topo-test" ]; then
    DATA_TSFRESH="$BASE_DIR/data/features/topo-test/selected"
    DATA_TOPO="$BASE_DIR/data/features/topo-test/topological_selected"
else
    DATA_TSFRESH="$BASE_DIR/data/features/dataset/selected"
    DATA_TOPO="$BASE_DIR/data/features/dataset/topological_selected"
fi

case "$FEATURES" in
    topo)
        DATA_SELECTED="$DATA_TOPO"
        OUTPUT_DIR="$BASE_DIR/03-models/flat_classifier/output_topo"
        EXTRA_PATH=""
        PREREQ="bash run-preprocessing.sh $MODE topo"
        ;;
    hybrid)
        DATA_SELECTED="$DATA_TSFRESH"
        OUTPUT_DIR="$BASE_DIR/03-models/flat_classifier/output_hybrid"
        EXTRA_PATH="$DATA_TOPO"
        PREREQ="bash run-preprocessing.sh $MODE topo"
        ;;
    *)  # tsfresh (default)
        DATA_SELECTED="$DATA_TSFRESH"
        OUTPUT_DIR="$BASE_DIR/03-models/flat_classifier/output"
        EXTRA_PATH=""
        PREREQ="bash run-preprocessing.sh $MODE"
        ;;
esac

[ -d "$DATA_SELECTED" ] || {
    echo "Error: $DATA_SELECTED not found."
    echo "  Run: $PREREQ"
    exit 1
}

if [ -n "$EXTRA_PATH" ]; then
    [ -d "$EXTRA_PATH" ] || {
        echo "Error: $EXTRA_PATH not found."
        echo "  Run: $PREREQ"
        exit 1
    }
fi

echo "============================================================"
echo "Training — $MODE  |  features: $FEATURES"
echo "============================================================"
echo "Input : $DATA_SELECTED"
[ -n "$EXTRA_PATH" ] && echo "Extra : $EXTRA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Start : $(date)"

mkdir -p "$OUTPUT_DIR"

TRAIN_CMD="$PYTHON 03-models/flat_classifier/train.py \
    --features-path $DATA_SELECTED \
    --save-dir      $OUTPUT_DIR \
    --n-jobs        -1"

[ -n "$EXTRA_PATH" ] && TRAIN_CMD="$TRAIN_CMD --extra-features-path $EXTRA_PATH"

$TRAIN_CMD

[ -f "$OUTPUT_DIR/classifier.pkl" ] || { echo "Error: Training failed."; exit 1; }

echo "============================================================"
echo "Training Complete!  features=$FEATURES"
echo "Output: $OUTPUT_DIR"
echo "End   : $(date)"
echo "============================================================"
